from typing import Any, List, Dict
import httpx
import json
from mcp.server.fastmcp import FastMCP

print("Starting Secure Analysis Server...")

# Initialize FastMCP server
mcp = FastMCP("secureanalysis")

# Constants - Update these to match your Flask backend
BACKEND_BASE_URL = "http://localhost:5000"  # Change this to your actual backend URL
REQUEST_TIMEOUT = 300.0  # 5 minutes timeout for analysis requests

# Available analysis categories
AVAILABLE_CATEGORIES = [
    "Grocery Retail (Default)",
    "Electrical Retail (Default)",
    "Business Improvement Districts"
]

async def make_backend_request(endpoint: str, data: dict = None, method: str = "POST") -> dict[str, Any] | None:
    """Make a request to the Flask backend with proper error handling."""
    url = f"{BACKEND_BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            if method == "POST":
                response = await client.post(
                    url, 
                    json=data, 
                    headers=headers, 
                    timeout=REQUEST_TIMEOUT
                )
            else:
                response = await client.get(
                    url, 
                    headers=headers, 
                    timeout=REQUEST_TIMEOUT
                )
            
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            return {"error": "Request timeout - analysis took too long"}
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                return {"error": f"HTTP {e.response.status_code}: {error_data.get('error', str(e))}"}
            except:
                return {"error": f"HTTP {e.response.status_code}: {str(e)}"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

def format_analysis_results(response_data: dict) -> str:
    """Format analysis results into a readable string."""
    if "error" in response_data:
        return f"Analysis Error: {response_data['error']}"
    
    if response_data.get("status") != "success":
        return f"Analysis failed: {response_data.get('message', 'Unknown error')}"
    
    # Format successful response
    result_text = f"""

Analysis Summary:
- Category: {response_data.get('category', 'Unknown')}
- Postcodes Processed: {response_data.get('postcodes_processed', 0)}
- Datasets Loaded: {response_data.get('datasets_loaded', 0)}
- S3 Bucket: {response_data.get('s3_bucket', 'Unknown')}
- Data Directory: {response_data.get('data_directory', 'Unknown')}

 Results Storage:
- S3 Key: {response_data.get('result_s3_key', 'Not saved')}
- Download URL: {'Available' if response_data.get('download_url') else 'Not available'}

 Analysis Results:
"""
    
    # Format the actual results
    results = response_data.get('results', {})
    
    if 'primary_results' in results and 'secondary_results' in results:
        # Two DataFrames returned
        primary_count = len(results['primary_results']) if results['primary_results'] else 0
        secondary_count = len(results['secondary_results']) if results['secondary_results'] else 0
        
        result_text += f"""

Primary Results:"""
        
        if results['primary_results'] and len(results['primary_results']) > 0:
            sample_primary = results['primary_results'][:3]  # Show first 3 records
            for i, record in enumerate(sample_primary, 1):
                result_text += f"\n  {i}. {json.dumps(record, indent=4)}"
        
        if results['secondary_results'] and len(results['secondary_results']) > 0:
            result_text += f"\n\n** Secondary Results:**"
            sample_secondary = results['secondary_results'][:3]  # Show first 3 records
            for i, record in enumerate(sample_secondary, 1):
                result_text += f"\n  {i}. {json.dumps(record, indent=4)}"
    
    elif 'results' in results:
        # Single DataFrame or other format
        if isinstance(results['results'], list):
            result_count = len(results['results'])
            result_text += f"""

Sample Results:"""
            
            if result_count > 0:
                sample_results = results['results'][:5]  # Show first 5 records
                for i, record in enumerate(sample_results, 1):
                    result_text += f"\n  {i}. {json.dumps(record, indent=4)}"
        else:
            result_text += f"\n- Results: {json.dumps(results['results'], indent=2)}"
    
    # Add download information if available
    if response_data.get('download_url'):
        result_text += f"""

Download Link:
{response_data['download_url']}
(Link expires in 1 hour)
"""
    
    return result_text

@mcp.tool()
async def analyze_postcodes(
    postcodes: List[str], 
    category: str = "Grocery Retail (Default)",
    bucket_name: str = "",
    data_directory: str = "Documents/DATA 2",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    region: str = "us-east-1"
) -> str:
    """Analyze postcodes using the secure analysis backend.

    Args:
        postcodes: List of UK postcodes to analyze (e.g., ["RG2 9UW", "HP2 4JW"])
        category: Analysis category - must be one of the available categories
        bucket_name: S3 bucket name where data is stored (required)
        data_directory: Directory path in S3 bucket where data files are located
        aws_access_key_id: AWS access key ID (optional if using IAM roles)
        aws_secret_access_key: AWS secret access key (optional if using IAM roles)
        region: AWS region (default: us-east-1)
    """
    
    # Validation
    if not postcodes:
        return " Error: No postcodes provided. Please provide a list of UK postcodes."
    
    if not bucket_name:
        return " Error: S3 bucket name is required. Please provide the bucket name where your data is stored."
    
    if category not in AVAILABLE_CATEGORIES:
        return f" Error: Invalid category '{category}'. Available categories:\n" + "\n".join([f"- {cat}" for cat in AVAILABLE_CATEGORIES])
    
    # Prepare request payload
    request_data = {
        "postcodes": postcodes,
        "category": category,
        "s3_config": {
            "bucket_name": bucket_name,
            "data_directory": data_directory,
            "region": region
        }
    }
    
    # Add AWS credentials if provided
    if aws_access_key_id:
        request_data["s3_config"]["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        request_data["s3_config"]["aws_secret_access_key"] = aws_secret_access_key
    
    # print(f"Analyzing {len(postcodes)} postcodes with category: {category}")
    
    # Make request to backend
    response = await make_backend_request("/analyze", request_data)
    
    if not response:
        return " Error: Unable to connect to the analysis backend. Please check if the backend service is running."
    
    return format_analysis_results(response)

# @mcp.tool()
# async def get_download_url(result_key: str, bucket_name: str, aws_access_key_id: str = "", aws_secret_access_key: str = "", region: str = "us-east-1") -> str:
#     """Generate a download URL for previously saved analysis results.

#     Args:
#         result_key: The S3 key of the result file (e.g., "results/analysis_20241201_143052_3_postcodes.json")
#         bucket_name: S3 bucket name where results are stored
#         aws_access_key_id: AWS access key ID (optional if using IAM roles)
#         aws_secret_access_key: AWS secret access key (optional if using IAM roles)
#         region: AWS region (default: us-east-1)
#     """
    
#     if not result_key or not bucket_name:
#         return " Error: Both result_key and bucket_name are required."
    
#     # Prepare request payload
#     request_data = {
#         "s3_config": {
#             "bucket_name": bucket_name,
#             "region": region
#         }
#     }
    
#     # Add AWS credentials if provided
#     if aws_access_key_id:
#         request_data["s3_config"]["aws_access_key_id"] = aws_access_key_id
#     if aws_secret_access_key:
#         request_data["s3_config"]["aws_secret_access_key"] = aws_secret_access_key
    
#     # Make request to backend
#     response = await make_backend_request(f"/download/{result_key}", request_data)
    
#     if not response:
#         return " Error: Unable to connect to the backend service."
    
#     if "error" in response:
#         return f" Error: {response['error']}"
    
#     return f"""
#  Download URL Generated

# Download Link:
# {response.get('download_url', 'URL not available')}

#  Expires: {response.get('expires_in', 'Unknown')}
#  Result Key: {response.get('result_key', result_key)}
# """


@mcp.tool()
async def analyze_postcodes_defaultdata(
    postcodes: List[str], 
    category: str = "Grocery Retail (Default)"
) -> str:
    """Get likelihood scores for postcodes using default datasets (no S3 configuration needed).

    Args:
        postcodes: List of UK postcodes to analyze (e.g., ["RG2 9UW", "HP2 4JW"])
        category: Analysis category - must be one of the available categories
    """
    
    # Validation
    if not postcodes:
        return " Error: No postcodes provided. Please provide a list of UK postcodes."
    
    if category not in AVAILABLE_CATEGORIES:
        return f"Error: Invalid category '{category}'. Available categories:\n" + "\n".join([f"- {cat}" for cat in AVAILABLE_CATEGORIES])
    
    # Prepare request payload
    request_data = {
        "postcode": postcodes,  # Note: backend expects 'postcode' not 'postcodes'
        "category": category
    }
    
    
    # Make request to backend
    response = await make_backend_request("/getlikelihoodscore", request_data)
    
    if not response:
        return "Error: Unable to connect to the analysis backend. Please check if the backend service is running."
    
    return format_likelihood_results(response)

def format_likelihood_results(response_data: dict) -> str:
    """Format likelihood score results into a readable string."""
    if "error" in response_data:
        return f"Likelihood Score Error: {response_data['error']}"
    
    if response_data.get("status") != "success":
        return f"Analysis failed: {response_data.get('error', 'Unknown error')}"
    
    data = response_data.get("data", {})
    
    # Format successful response
    result_text = f"""
**Likelihood Score Analysis Complete**

📊 **Analysis Summary:**
- Postcodes: {', '.join(data.get('postcode', []))}
- Category: {data.get('category', 'Unknown')}
- DataFrame 1 Records: {data.get('dataframe1_shape', [0, 0])[0]}
- DataFrame 2 Records: {data.get('dataframe2_shape', [0, 0])[0]}

📈 **Results Overview:**
- DataFrame 1 Shape: {data.get('dataframe1_shape', 'Unknown')} (rows, columns)
- DataFrame 2 Shape: {data.get('dataframe2_shape', 'Unknown')} (rows, columns)

📋 **Detailed Results:**
"""
    
    # Format DataFrame 1 results
    df1_data = data.get('dataframe1', [])
    if df1_data:
        result_text += f"\n**📊 Primary Analysis Results ({len(df1_data)} records):**"
        
        # Show first few records
        sample_size = min(5, len(df1_data))
        for i, record in enumerate(df1_data[:sample_size], 1):
            result_text += f"\n\n  Record {i}:"
            for key, value in record.items():
                # Format numeric values nicely
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    result_text += f"\n    • {key}: {formatted_value}"
                else:
                    result_text += f"\n    • {key}: {value}"
        
        if len(df1_data) > sample_size:
            result_text += f"\n\n  ... and {len(df1_data) - sample_size} more records"
    else:
        result_text += "\n**📊 Primary Analysis Results:** No data returned"
    
    # Format DataFrame 2 results
    df2_data = data.get('dataframe2', [])
    if df2_data:
        result_text += f"\n\n**📈 Secondary Analysis Results ({len(df2_data)} records):**"
        
        # Show first few records
        sample_size = min(3, len(df2_data))
        for i, record in enumerate(df2_data[:sample_size], 1):
            result_text += f"\n\n  Record {i}:"
            for key, value in record.items():
                # Format numeric values nicely
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    result_text += f"\n    • {key}: {formatted_value}"
                else:
                    result_text += f"\n    • {key}: {value}"
        
        if len(df2_data) > sample_size:
            result_text += f"\n\n  ... and {len(df2_data) - sample_size} more records"
    else:
        result_text += "\n\n**📈 Secondary Analysis Results:** No data returned"
    
    # Add summary statistics if numeric data is present
    if df1_data:
        result_text += "\n\n💡 **Quick Insights:**"
        
        # Look for common likelihood/score columns
        numeric_columns = []
        if df1_data:
            sample_record = df1_data[0]
            numeric_columns = [k for k, v in sample_record.items() 
                             if isinstance(v, (int, float)) and 
                             any(keyword in k.lower() for keyword in ['score', 'likelihood', 'probability', 'rating'])]
        
        if numeric_columns:
            for col in numeric_columns[:3]:  # Show up to 3 key metrics
                values = [record.get(col, 0) for record in df1_data if isinstance(record.get(col), (int, float))]
                if values:
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    result_text += f"\n  • {col}: Avg={avg_val:.3f}, Max={max_val:.3f}, Min={min_val:.3f}"
    
    return result_text


@mcp.tool()
async def list_available_categories() -> str:
    """List all available analysis categories."""
    
    categories_text = " Available  Categories:\n\n"
    for i, category in enumerate(AVAILABLE_CATEGORIES, 1):
        categories_text += f"{i}. {category}\n"
    
    categories_text += "\n Usage: Use any of these categories in the `analyze_postcodes` function."
    
    return categories_text

@mcp.tool()
async def get_backend_status() -> str:
    """Check if the analysis backend is running and accessible."""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_BASE_URL}/", timeout=10.0)
            if response.status_code == 404:
                return f" Backend is running at {BACKEND_BASE_URL} (Flask default 404 response)"
            else:
                return f" Backend is accessible at {BACKEND_BASE_URL} (Status: {response.status_code})"
    except Exception as e:
        return f" Backend is not accessible at {BACKEND_BASE_URL}\nError: {str(e)}\n\nPlease ensure your Flask backend is running."

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')