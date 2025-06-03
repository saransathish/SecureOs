import logging
import pandas as pd
import os
import json
import boto3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pyarrow.csv as pv
from flask import Flask, request, jsonify
from botocore.exceptions import ClientError, NoCredentialsError
import io
import asyncio
import aiohttp
from multiprocessing import cpu_count, Manager, Pool
import threading
import queue
import time
from functools import partial
import numpy as np
from final_score_weightings import final_weightings as fsw
from model import run_model

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global configuration for parallel processing
MAX_WORKERS = min(cpu_count() * 2, 20)  # Optimal thread count
MAX_PROCESS_WORKERS = cpu_count()       # Process count for CPU-intensive tasks
CHUNK_SIZE = 100                        # Batch size for processing

class OptimizedS3DataManager:
    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-1'):
        """
        Initialize S3 client with connection pooling for better performance
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self._lock = threading.Lock()
        
        # Create session for connection pooling
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        try:
            # Create multiple S3 clients for parallel operations
            self.s3_clients = []
            for _ in range(MAX_WORKERS):
                if aws_access_key_id and aws_secret_access_key:
                    client = boto3.client(
                        's3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name
                    )
                else:
                    client = boto3.client('s3', region_name=region_name)
                self.s3_clients.append(client)
            
            # Test connection with first client
            self.s3_clients[0].head_bucket(Bucket=bucket_name)
            logging.info(f"Successfully connected to S3 bucket: {bucket_name} with {len(self.s3_clients)} clients")
            
        except Exception as e:
            logging.error(f"Failed to connect to S3: {str(e)}")
            raise

    def get_s3_client(self):
        """Get an available S3 client from the pool"""
        with self._lock:
            return self.s3_clients[threading.current_thread().ident % len(self.s3_clients)]

    def load_file_from_s3_optimized(self, s3_key, client_index=0):
        """Optimized file loading with specific client"""
        try:
            logging.info(f"Loading file from S3: {s3_key}")
            
            # Use specific client to avoid conflicts
            s3_client = self.s3_clients[client_index % len(self.s3_clients)]
            
            # Get the object from S3
            response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            file_content = response['Body'].read()
            
            # Determine file type and load accordingly with optimizations
            file_extension = os.path.splitext(s3_key)[1].lower()
            
            if file_extension == '.csv':
                # Use PyArrow for faster CSV reading
                try:
                    df = pv.read_csv(io.BytesIO(file_content)).to_pandas()
                except:
                    df = pd.read_csv(io.BytesIO(file_content))
            elif file_extension == '.xlsx':
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            elif file_extension == '.ods':
                df = pd.read_excel(io.BytesIO(file_content), engine='odf')
            else:
                logging.error(f"Unsupported file format: {file_extension}")
                return None
            
            logging.info(f"Successfully loaded {s3_key}. Shape: {df.shape}")
            return df
            
        except ClientError as e:
            logging.error(f"Error loading {s3_key} from S3: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error processing {s3_key}: {str(e)}")
            return None

    def save_results_to_s3_parallel(self, results, s3_key):
        """Parallel save with chunking for large results"""
        try:
            # Handle different result types
            if isinstance(results, tuple) and len(results) == 2:
                df1, df2 = results
                results_json = {
                    "primary_results": df1.to_dict('records') if hasattr(df1, 'to_dict') else df1,
                    "secondary_results": df2.to_dict('records') if hasattr(df2, 'to_dict') else df2,
                    "metadata": {
                        "primary_shape": list(df1.shape) if hasattr(df1, 'shape') else None,
                        "secondary_shape": list(df2.shape) if hasattr(df2, 'shape') else None,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            elif isinstance(results, pd.DataFrame):
                results_json = {
                    "results": results.to_dict('records'),
                    "metadata": {
                        "shape": list(results.shape),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                results_json = {
                    "results": results,
                    "metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Use ujson for faster JSON serialization if available
            try:
                import ujson
                json_string = ujson.dumps(results_json, indent=2)
            except ImportError:
                json_string = json.dumps(results_json, indent=2, default=str)
            
            # Upload to S3 with first available client
            s3_client = self.get_s3_client()
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_string,
                ContentType='application/json'
            )
            
            logging.info(f"Results saved to S3: {s3_key}")
            return True, results_json
            
        except Exception as e:
            logging.error(f"Error saving results to S3: {str(e)}")
            return False, None

    def batch_operations(self, operations, batch_size=10):
        """Execute S3 operations in batches"""
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Process operations in batches
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i+batch_size]
                batch_futures = []
                
                for op in batch:
                    future = executor.submit(op['function'], *op['args'], **op['kwargs'])
                    batch_futures.append(future)
                
                # Collect batch results
                for future in as_completed(batch_futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Batch operation failed: {str(e)}")
                        results.append(None)
        
        return results

    def generate_presigned_url(self, s3_key, expiration=3600):
        """Generate presigned URL with first available client"""
        try:
            s3_client = self.get_s3_client()
            response = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            logging.error(f"Error generating presigned URL: {str(e)}")
            return None


def create_optimized_s3_manager(bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region='us-east-1'):
    """Create optimized S3 manager with connection pooling"""
    return OptimizedS3DataManager(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )


def parallel_dataframe_processing(df, processing_func, chunk_size=CHUNK_SIZE):
    """Process DataFrame in parallel chunks"""
    if len(df) <= chunk_size:
        return processing_func(df)
    
    # Split DataFrame into chunks
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=MAX_PROCESS_WORKERS) as executor:
        processed_chunks = list(executor.map(processing_func, chunks))
    
    # Combine results
    if isinstance(processed_chunks[0], pd.DataFrame):
        return pd.concat(processed_chunks, ignore_index=True)
    else:
        return processed_chunks


def load_datasets_from_s3_parallel(s3_manager, data_source_mapping):
    """Highly optimized parallel dataset loading"""
    dataframe_library = {}
    total_files = len(data_source_mapping)
    
    logging.info(f"Starting parallel load of {total_files} datasets from S3")
    start_time = time.time()
    
    def load_single_file_with_index(args):
        name, s3_key, client_index = args
        df = s3_manager.load_file_from_s3_optimized(s3_key, client_index)
        return name, df
    
    # Prepare arguments with client indices for load balancing
    file_args = [(name, s3_key, idx) for idx, (name, s3_key) in enumerate(data_source_mapping.items())]
    
    # Use optimal number of workers
    max_workers = min(MAX_WORKERS, total_files)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(load_single_file_with_index, args): args[0]
            for args in file_args
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                name, df = future.result()
                if df is not None and not df.empty:
                    dataframe_library[name] = df
                    completed += 1
                    logging.info(f"✓ Loaded {name} ({completed}/{total_files}). Shape: {df.shape}")
                else:
                    logging.warning(f"✗ Dataset {name} is empty or failed to load")
            except Exception as e:
                logging.error(f"✗ Error processing {name}: {str(e)}")
    
    load_time = time.time() - start_time
    logging.info(f"Parallel loading completed in {load_time:.2f}s. Loaded {len(dataframe_library)}/{total_files} datasets")
    
    return dataframe_library


def parallel_model_execution(data_library, postcodes_df, weightings):
    """Execute model with parallel processing optimizations"""
    
    def chunked_model_run(chunk_data):
        """Run model on a chunk of data"""
        return run_model(chunk_data['data_library'], chunk_data['postcodes'], chunk_data['weightings'])
    
    # If postcodes are too many, split them into chunks
    if len(postcodes_df) > CHUNK_SIZE:
        logging.info(f"Splitting {len(postcodes_df)} postcodes into chunks for parallel processing")
        
        postcode_chunks = [
            postcodes_df[i:i+CHUNK_SIZE] 
            for i in range(0, len(postcodes_df), CHUNK_SIZE)
        ]
        
        # Prepare chunk data
        chunk_data_list = [
            {
                'data_library': data_library,
                'postcodes': chunk,
                'weightings': weightings
            }
            for chunk in postcode_chunks
        ]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=MAX_PROCESS_WORKERS) as executor:
            chunk_results = list(executor.map(chunked_model_run, chunk_data_list))
        
        # Combine results (assuming results are DataFrames)
        if chunk_results:
            if isinstance(chunk_results[0], tuple):
                # Handle tuple results (primary, secondary)
                primary_results = []
                secondary_results = []
                for primary, secondary in chunk_results:
                    if isinstance(primary, pd.DataFrame):
                        primary_results.append(primary)
                    if isinstance(secondary, pd.DataFrame):
                        secondary_results.append(secondary)
                
                combined_primary = pd.concat(primary_results, ignore_index=True) if primary_results else pd.DataFrame()
                combined_secondary = pd.concat(secondary_results, ignore_index=True) if secondary_results else pd.DataFrame()
                
                return combined_primary, combined_secondary
            else:
                # Handle single DataFrame results
                if isinstance(chunk_results[0], pd.DataFrame):
                    return pd.concat(chunk_results, ignore_index=True)
                else:
                    return chunk_results
    else:
        # Run model directly for small datasets
        return run_model(data_library, postcodes_df, weightings)


def convert_results_to_json_parallel(results):
    """Convert model results to JSON format with parallel processing"""
    
    def parallel_dict_conversion(df_chunk):
        """Convert DataFrame chunk to dict in parallel"""
        return df_chunk.to_dict('records')
    
    if isinstance(results, tuple) and len(results) == 2:
        df1, df2 = results
        
        # Convert large DataFrames in parallel
        if hasattr(df1, 'to_dict') and len(df1) > CHUNK_SIZE:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(parallel_dict_conversion, df1)
                future2 = executor.submit(parallel_dict_conversion, df2)
                
                primary_dict = future1.result()
                secondary_dict = future2.result()
        else:
            primary_dict = df1.to_dict('records') if hasattr(df1, 'to_dict') else df1
            secondary_dict = df2.to_dict('records') if hasattr(df2, 'to_dict') else df2
        
        return {
            "primary_results": primary_dict,
            "secondary_results": secondary_dict,
            "result_type": "dual_dataframes",
            "primary_columns": list(df1.columns) if hasattr(df1, 'columns') else [],
            "secondary_columns": list(df2.columns) if hasattr(df2, 'columns') else [],
            "primary_count": len(df1) if hasattr(df1, '__len__') else 0,
            "secondary_count": len(df2) if hasattr(df2, '__len__') else 0
        }
    elif isinstance(results, pd.DataFrame):
        # Handle large single DataFrame
        if len(results) > CHUNK_SIZE:
            records = parallel_dict_conversion(results)
        else:
            records = results.to_dict('records')
        
        return {
            "results": records,
            "result_type": "single_dataframe",
            "columns": list(results.columns),
            "count": len(results)
        }
    else:
        return {
            "results": results,
            "result_type": "other"
        }


# Default data source mapping (unchanged)
default_data_source_mapping = {
    "uk_monthly_police_reporting": "CY Crime/sector_level_crime.csv",
    "schools": "Schools/schools_for_script.csv",
    "uk_retail_parks_&_shopping_centres": "Retail & Shopping Parks/retail_shopping_park_locations.csv",
    "uk_major_junction": "Junctions/Junctions.xlsx",
    "uk_transport_hubs": "Transport Hubs/England_Scotland_Wales_transport_hubs.csv",
    "uk_wellbeing": "Wellbeing/uk_wellbeing_for_script.csv",
    "uk_unemployment": "Unemployment_Student/uk_unemployment_for_script.csv",
    "uk_historic_detailed_police_reporting": "Historic Crime/uk_historic_crime_data_for_script.xlsx",
    "scottish_monthly_police_reporting": "CY Crime/Police Scotland/Scotland_CY_Crime.csv",
    "uk_population_density": "Population Density/uk_population_density_for_script.csv",
    "scotland_population_density": "Population Density/scotland_population_density_for_script.csv",
    "uk_student_population": "Unemployment_Student/uk_student_population_for_script.csv",
    "scotland_student_population": "Unemployment_Student/scotland_student_population_for_script.csv",
    "scotland_unemployment": "Unemployment_Student/scotland_unemployment_for_script.csv",
    "scotland_historic_detailed_police_reporting": "Historic Crime/scotland_historic_crime_data_for_script.csv",
    "ni_monthly_police_reporting": "CY Crime/ni_monthly_crime_for_script.csv",
    "ni_population_density": "Population Density/ni_population_density_for_script.csv",
    "ni_wellbeing": "Wellbeing/ni_wellbeing_for_script.csv",
    "ni_student_population": "Unemployment_Student/ni_student_population_for_script.csv",
    "ni_unemployment": "Unemployment_Student/ni_unemployment_for_script.csv",
    "pubs": "Pubs/pubs_for_script.csv",
    "scottish_postcode_directory": "Historic Crime/scottish_postcode_directory.csv",
    "population": "Postcodes/ukpostcodes.csv",
    "uk_homelessness": "Homelessness/uk_homelessness_for_script.csv",
    "scotland_homelessness": "Homelessness/scotland_homelessness_for_script.csv",
    "ni_homelessness": "Homelessness/ni_homelessness_for_script.csv"
}


@app.route('/analyze', methods=['POST'])
def analyze_postcodes_optimized():
    """
    Highly optimized API endpoint with parallel processing at all levels
    """
    overall_start_time = time.time()
    
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        postcodes_list = data.get('postcodes', [])
        category = data.get('category', 'Grocery Retail (Default)')
        s3_config = data.get('s3_config', {})
        
        if not postcodes_list:
            return jsonify({"error": "No postcodes provided"}), 400
        
        if category not in fsw:
            return jsonify({
                "error": f"Invalid category. Available categories: {list(fsw.keys())}"
            }), 400
        
        if not s3_config.get('bucket_name'):
            return jsonify({"error": "S3 bucket_name is required in s3_config"}), 400
        
        # Convert postcodes to DataFrame with parallel processing if large
        if len(postcodes_list) > CHUNK_SIZE:
            postcodes_df = pd.DataFrame(postcodes_list, columns=['postcode'])
        else:
            postcodes_df = pd.DataFrame(postcodes_list, columns=['postcode'])
        
        logging.info(f"🚀 Starting optimized processing of {len(postcodes_list)} postcodes with category: {category}")
        logging.info(f"📁 Using S3 bucket: {s3_config['bucket_name']}")
        
        # Phase 1: Create optimized S3 manager
        phase1_start = time.time()
        s3_manager = create_optimized_s3_manager(
            bucket_name=s3_config['bucket_name'],
            aws_access_key_id=s3_config.get('aws_access_key_id'),
            aws_secret_access_key=s3_config.get('aws_secret_access_key'),
            region=s3_config.get('region', 'us-east-1')
        )
        phase1_time = time.time() - phase1_start
        logging.info(f"⚡ Phase 1 (S3 Setup): {phase1_time:.2f}s")
        
        # Phase 2: Build data source mapping
        phase2_start = time.time()
        data_directory = s3_config.get('data_directory', 'Documents/DATA 2')
        current_mapping = {
            name: f"{data_directory}/{path}" 
            for name, path in default_data_source_mapping.items()
        }
        phase2_time = time.time() - phase2_start
        logging.info(f"⚡ Phase 2 (Mapping): {phase2_time:.2f}s")
        
        # Phase 3: Parallel dataset loading
        phase3_start = time.time()
        data_library = load_datasets_from_s3_parallel(s3_manager, current_mapping)
        phase3_time = time.time() - phase3_start
        logging.info(f"⚡ Phase 3 (Data Loading): {phase3_time:.2f}s")
        
        if not data_library:
            return jsonify({"error": "Failed to load any datasets from S3"}), 500
        
        # Phase 4: Parallel model execution
        phase4_start = time.time()
        logging.info("🔄 Running parallel model analysis...")
        model_results = parallel_model_execution(data_library, postcodes_df, fsw[category])
        phase4_time = time.time() - phase4_start
        logging.info(f"⚡ Phase 4 (Model Execution): {phase4_time:.2f}s")
        
        # Phase 5: Parallel result conversion
        phase5_start = time.time()
        json_results = convert_results_to_json_parallel(model_results)
        phase5_time = time.time() - phase5_start
        logging.info(f"⚡ Phase 5 (Result Conversion): {phase5_time:.2f}s")
        
        # Phase 6: Parallel S3 save and URL generation
        phase6_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_key = f"results/analysis_{timestamp}_{len(postcodes_list)}_postcodes.json"
        
        # Use ThreadPoolExecutor for parallel S3 operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            save_future = executor.submit(s3_manager.save_results_to_s3_parallel, model_results, result_key)
            
            # Wait for save to complete, then generate URL
            save_success, saved_results = save_future.result()
            
            # Generate download URL
            download_url = None
            if save_success:
                download_url = s3_manager.generate_presigned_url(result_key)
        
        phase6_time = time.time() - phase6_start
        logging.info(f"⚡ Phase 6 (S3 Save & URL): {phase6_time:.2f}s")
        
        # Calculate total processing time
        total_time = time.time() - overall_start_time
        
        # Prepare optimized response
        response_data = {
            "status": "success",
            "message": f"✅ Optimized analysis completed for {len(postcodes_list)} postcodes",
            "category": category,
            "postcodes_processed": len(postcodes_list),
            "datasets_loaded": len(data_library),
            "s3_bucket": s3_config['bucket_name'],
            "data_directory": data_directory,
            "result_s3_key": result_key if save_success else None,
            "download_url": download_url,
            "performance_metrics": {
                "total_time_seconds": round(total_time, 2),
                "phase_times": {
                    "s3_setup": round(phase1_time, 2),
                    "mapping": round(phase2_time, 4),
                    "data_loading": round(phase3_time, 2),
                    "model_execution": round(phase4_time, 2),
                    "result_conversion": round(phase5_time, 2),
                    "s3_save_url": round(phase6_time, 2)
                },
                "parallel_workers": {
                    "max_thread_workers": MAX_WORKERS,
                    "max_process_workers": MAX_PROCESS_WORKERS,
                    "s3_clients": len(s3_manager.s3_clients)
                }
            },
            "results": json_results
        }
        
        logging.info(f"🎉 Total optimized processing time: {total_time:.2f}s")
        return jsonify(response_data), 200
        
    except Exception as e:
        logging.error(f"❌ Error in optimized analyze_postcodes: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "An error occurred during optimized analysis"
        }), 500


@app.route('/download/<path:result_key>', methods=['POST'])
def download_result_optimized(result_key):
    """Optimized download URL generation"""
    try:
        data = request.get_json()
        s3_config = data.get('s3_config', {})
        
        if not s3_config.get('bucket_name'):
            return jsonify({"error": "bucket_name is required"}), 400
        
        # Create optimized S3 manager
        s3_manager = create_optimized_s3_manager(
            bucket_name=s3_config['bucket_name'],
            aws_access_key_id=s3_config.get('aws_access_key_id'),
            aws_secret_access_key=s3_config.get('aws_secret_access_key'),
            region=s3_config.get('region', 'us-east-1')
        )
        
        download_url = s3_manager.generate_presigned_url(result_key)
        if download_url:
            return jsonify({
                "download_url": download_url,
                "expires_in": "1 hour",
                "result_key": result_key
            }), 200
        else:
            return jsonify({"error": "Failed to generate download URL"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system info"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cpu_count": cpu_count(),
            "max_thread_workers": MAX_WORKERS,
            "max_process_workers": MAX_PROCESS_WORKERS,
            "chunk_size": CHUNK_SIZE
        }
    }), 200


if __name__ == '__main__':
    # Set optimal Flask configuration for production
    app.run(
        debug=False,  # Disable debug for production performance
        host='0.0.0.0', 
        port=4000,
        threaded=True,  # Enable threading
        processes=1    # Use threading instead of processes for Flask
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    