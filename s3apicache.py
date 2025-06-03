import logging
import pandas as pd
import os
import json
import boto3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.csv as pv
from flask import Flask, request, jsonify
from botocore.exceptions import ClientError, NoCredentialsError
import io
import hashlib
import threading
from collections import OrderedDict
from final_score_weightings import final_weightings as fsw
from model import run_model
import cProfile

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

class LRUCache:
    """Thread-safe LRU Cache implementation"""
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def size(self):
        with self.lock:
            return len(self.cache)

class DatasetCache:
    """Cache for storing loaded datasets with expiration"""
    def __init__(self, expiration_hours=24):
        self.cache = {}
        self.timestamps = {}
        self.expiration_delta = timedelta(hours=expiration_hours)
        self.lock = threading.Lock()
    
    def get(self, cache_key):
        with self.lock:
            if cache_key not in self.cache:
                return None
            
            # Check if expired
            if datetime.now() - self.timestamps[cache_key] > self.expiration_delta:
                del self.cache[cache_key]
                del self.timestamps[cache_key]
                logging.info(f"Cache expired for key: {cache_key}")
                return None
            
            logging.info(f"Cache hit for datasets: {cache_key}")
            return self.cache[cache_key]
    
    def put(self, cache_key, data):
        with self.lock:
            self.cache[cache_key] = data
            self.timestamps[cache_key] = datetime.now()
            logging.info(f"Cached datasets with key: {cache_key}")
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_cache_stats(self):
        with self.lock:
            total_size = len(self.cache)
            expired_count = sum(1 for ts in self.timestamps.values() 
                              if datetime.now() - ts > self.expiration_delta)
            return {
                "total_entries": total_size,
                "expired_entries": expired_count,
                "active_entries": total_size - expired_count
            }

# Global cache instances
dataset_cache = DatasetCache(expiration_hours=24)  # Datasets cached for 24 hours
postcode_results_cache = LRUCache(capacity=1000)   # Results cached for 1000 most recent queries

class S3DataManager:
    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-1'):
        """
        Initialize S3 client
        If credentials are not provided, boto3 will use default credential chain
        (environment variables, IAM roles, etc.)
        """
        self.bucket_name = bucket_name
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                self.s3_client = boto3.client('s3', region_name=region_name)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"Successfully connected to S3 bucket: {bucket_name}")
        except Exception as e:
            logging.error(f"Failed to connect to S3: {str(e)}")
            raise

    def load_file_from_s3(self, s3_key):
        """Load a file from S3 and return as pandas DataFrame"""
        try:
            logging.info(f"Loading file from S3: {s3_key}")
            
            # Get the object from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            file_content = response['Body'].read()
            
            # Determine file type and load accordingly
            file_extension = os.path.splitext(s3_key)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif file_extension == '.xlsx':
                df = pd.read_excel(io.BytesIO(file_content))
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

    def save_results_to_s3(self, results, s3_key):
        """Save results to S3 as JSON - handles multiple DataFrames"""
        try:
            # Handle different result types
            if isinstance(results, tuple) and len(results) == 2:
                # Two DataFrames returned from run_model
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
                # Single DataFrame
                results_json = {
                    "results": results.to_dict('records'),
                    "metadata": {
                        "shape": list(results.shape),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                # Other format
                results_json = {
                    "results": results,
                    "metadata": {
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Convert to JSON string
            json_string = json.dumps(results_json, indent=2, default=str)
            
            # Upload to S3
            self.s3_client.put_object(
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

    def generate_presigned_url(self, s3_key, expiration=3600):
        """Generate a presigned URL for downloading the results"""
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            logging.error(f"Error generating presigned URL: {str(e)}")
            return None

def generate_cache_key(s3_config, data_directory):
    """Generate a cache key for dataset loading based on S3 config"""
    key_components = [
        s3_config.get('bucket_name', ''),
        data_directory,
        s3_config.get('region', 'us-east-1')
    ]
    key_string = '|'.join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()

def generate_postcode_cache_key(postcodes, category):
    """Generate a cache key for postcode analysis results"""
    # Sort postcodes to ensure consistent cache keys
    sorted_postcodes = sorted(postcodes)
    key_components = [
        ','.join(sorted_postcodes),
        category
    ]
    key_string = '|'.join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()

def create_s3_manager(bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region='us-east-1'):
    """Create S3 manager with provided credentials"""
    return S3DataManager(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

def load_datasets_from_s3_with_cache(s3_manager, data_source_mapping, cache_key):
    """Load all datasets from S3 with caching support"""
    # Check cache first
    cached_data = dataset_cache.get(cache_key)
    if cached_data is not None:
        logging.info("Using cached datasets")
        return cached_data
    
    # Load from S3 if not in cache
    logging.info("Loading datasets from S3 (not in cache)")
    dataframe_library = {}
    total_files = len(data_source_mapping)
    
    logging.info(f"Starting to load {total_files} datasets from S3")
    
    def load_single_file(name, s3_key):
        df = s3_manager.load_file_from_s3(s3_key)
        return name, df
    
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        future_to_name = {
            executor.submit(load_single_file, name, s3_key): name
            for name, s3_key in data_source_mapping.items()
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                name, df = future.result()
                if df is not None and not df.empty:
                    dataframe_library[name] = df
                    logging.info(f"Successfully loaded {name}. Shape: {df.shape}")
                else:
                    logging.warning(f"Dataset {name} is empty or failed to load")
            except Exception as e:
                logging.error(f"Error processing {name}: {str(e)}")
    
    # Cache the loaded data
    if dataframe_library:
        dataset_cache.put(cache_key, dataframe_library)
    
    return dataframe_library

def check_postcode_subset_cache(requested_postcodes, category):
    """Check if we have cached results that contain all requested postcodes"""
    requested_set = set(requested_postcodes)
    
    # This is a simplified approach - in practice, you might want to store
    # metadata about which postcodes are in each cached result
    for i in range(postcode_results_cache.size()):
        # In a more sophisticated implementation, you'd store postcode sets
        # alongside results and check for supersets
        pass
    
    return None

def process_postcodes_with_cache(data_library, postcodes_list, category):
    """Process postcodes with caching support"""
    # Generate cache key for this specific request
    postcode_cache_key = generate_postcode_cache_key(postcodes_list, category)
    
    # Check if exact results are cached
    cached_results = postcode_results_cache.get(postcode_cache_key)
    if cached_results is not None:
        logging.info(f"Using cached results for {len(postcodes_list)} postcodes")
        return cached_results, True  # True indicates cache hit
    
    # Run the model for new postcodes
    postcodes_df = pd.DataFrame(postcodes_list, columns=['postcode'])
    logging.info(f"Running model analysis for {len(postcodes_list)} postcodes (not in cache)")
    
    model_results = run_model(data_library, postcodes_df, fsw[category])
    
    # Cache the results
    postcode_results_cache.put(postcode_cache_key, model_results)
    
    return model_results, False  # False indicates no cache hit

def convert_results_to_json_format(results):
    """Convert model results to JSON serializable format"""
    if isinstance(results, tuple) and len(results) == 2:
        # Two DataFrames returned
        df1, df2 = results
        return {
            "primary_results": df1.to_dict('records') if hasattr(df1, 'to_dict') else df1,
            "secondary_results": df2.to_dict('records') if hasattr(df2, 'to_dict') else df2,
            "result_type": "dual_dataframes",
            "primary_columns": list(df1.columns) if hasattr(df1, 'columns') else [],
            "secondary_columns": list(df2.columns) if hasattr(df2, 'columns') else [],
            "primary_count": len(df1) if hasattr(df1, '__len__') else 0,
            "secondary_count": len(df2) if hasattr(df2, '__len__') else 0
        }
    elif isinstance(results, pd.DataFrame):
        # Single DataFrame
        return {
            "results": results.to_dict('records'),
            "result_type": "single_dataframe",
            "columns": list(results.columns),
            "count": len(results)
        }
    else:
        # Other format
        return {
            "results": results,
            "result_type": "other"
        }

# Default data source mapping (relative paths)
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
def analyze_postcodes():
    """
    Main API endpoint for postcode analysis with caching
    Expected JSON payload:
    {
        "postcodes": ["1011 AB", "RG2 9UW", "HP2 4JW"],
        "category": "Grocery Retail (Default)",
        "s3_config": {
            "bucket_name": "your-bucket-name",
            "data_directory": "Documents/DATA 2",
            "aws_access_key_id": "optional",
            "aws_secret_access_key": "optional",
            "region": "us-east-1"
        }
    }
    """
    try:
        start_time = datetime.now()
        
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
        
        logging.info(f"Processing {len(postcodes_list)} postcodes with category: {category}")
        logging.info(f"Using S3 bucket: {s3_config['bucket_name']}")
        
        # Generate cache keys
        data_directory = s3_config.get('data_directory', 'Documents/DATA 2')
        dataset_cache_key = generate_cache_key(s3_config, data_directory)
        
        # Create S3 manager
        s3_manager = create_s3_manager(
            bucket_name=s3_config['bucket_name'],
            aws_access_key_id=s3_config.get('aws_access_key_id'),
            aws_secret_access_key=s3_config.get('aws_secret_access_key'),
            region=s3_config.get('region', 'us-east-1')
        )
        
        # Build data source mapping with provided directory
        current_mapping = {
            name: f"{data_directory}/{path}" 
            for name, path in default_data_source_mapping.items()
        }
        
        # Load datasets with caching
        data_library = load_datasets_from_s3_with_cache(s3_manager, current_mapping, dataset_cache_key)
        
        if not data_library:
            return jsonify({"error": "Failed to load any datasets from S3"}), 500
        
        dataset_load_time = datetime.now()
        
        # Process postcodes with caching
        model_results, was_cached = process_postcodes_with_cache(data_library, postcodes_list, category)
        
        analysis_time = datetime.now()
        
        # Convert results to JSON format
        json_results = convert_results_to_json_format(model_results)
        
        # Generate unique result key for S3 (only if not from cache)
        result_key = None
        download_url = None
        save_success = False
        
        if not was_cached:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_key = f"results/analysis_{timestamp}_{len(postcodes_list)}_postcodes.json"
            
            # Save results to S3
            save_success, saved_results = s3_manager.save_results_to_s3(model_results, result_key)
            
            # Generate presigned URL for result download
            if save_success:
                download_url = s3_manager.generate_presigned_url(result_key)
        
        end_time = datetime.now()
        
        # Calculate timing metrics
        total_time = (end_time - start_time).total_seconds()
        dataset_load_time_sec = (dataset_load_time - start_time).total_seconds()
        analysis_time_sec = (analysis_time - dataset_load_time).total_seconds()
        
        # Get cache statistics
        cache_stats = dataset_cache.get_cache_stats()
        
        # Prepare response with performance metrics
        response_data = {
            "status": "success",
            "message": f"Analysis completed for {len(postcodes_list)} postcodes",
            "category": category,
            "postcodes_processed": len(postcodes_list),
            "datasets_loaded": len(data_library),
            "s3_bucket": s3_config['bucket_name'],
            "data_directory": data_directory,
            # "result_s3_key": result_key if save_success else None,
            "download_url": download_url,
            "results": json_results,
            # "performance_metrics": {
            #     "total_time_seconds": round(total_time, 2),
            #     "dataset_load_time_seconds": round(dataset_load_time_sec, 2),
            #     "analysis_time_seconds": round(analysis_time_sec, 2),
            #     "used_cached_datasets": dataset_cache.get(dataset_cache_key) is not None,
            #     "used_cached_results": was_cached
            # },
            # "cache_statistics": {
            #     "dataset_cache": cache_stats,
            #     "postcode_results_cache_size": postcode_results_cache.size()
            # }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logging.error(f"Error in analyze_postcodes: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "An error occurred during analysis"
        }), 500

@app.route('/cache/status', methods=['GET'])
def get_cache_status():
    """Get current cache status and statistics"""
    try:
        dataset_stats = dataset_cache.get_cache_stats()
        
        return jsonify({
            "status": "success",
            "cache_statistics": {
                "dataset_cache": {
                    **dataset_stats,
                    "expiration_hours": 24
                },
                "postcode_results_cache": {
                    "current_size": postcode_results_cache.size(),
                    "max_capacity": postcode_results_cache.capacity
                }
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Error getting cache status: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches"""
    try:
        data = request.get_json() or {}
        cache_type = data.get('cache_type', 'all')  # 'all', 'datasets', 'postcodes'
        
        if cache_type in ['all', 'datasets']:
            dataset_cache.clear()
            logging.info("Dataset cache cleared")
        
        if cache_type in ['all', 'postcodes']:
            postcode_results_cache.clear()
            logging.info("Postcode results cache cleared")
        
        return jsonify({
            "status": "success",
            "message": f"Cache cleared: {cache_type}",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/download/<path:result_key>', methods=['POST'])
def download_result(result_key):
    """Generate download URL for a specific result"""
    try:
        data = request.get_json()
        s3_config = data.get('s3_config', {})
        
        if not s3_config.get('bucket_name'):
            return jsonify({"error": "bucket_name is required"}), 400
        
        # Create S3 manager
        s3_manager = create_s3_manager(
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)