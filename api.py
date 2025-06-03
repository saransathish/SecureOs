# from model import Api

# print('Running API... call')
# print( "output recived in api" , Api(['RG2 9UW'] , 'Grocery Retail (Default)' ))
# print('API runed...')


# from flask import Flask, request, jsonify
# from model import Api

# app = Flask(__name__)

# @app.route('/getlikelihoodscore', methods=['POST'])
# def get_likelihood_score():
#     print('API endpoint hit')
#     try:
#         data = request.get_json()
        
#         # Extract values from request
#         postcode = data.get('postcode')
#         category = data.get('category')

#         if not postcode or not category:
#             return jsonify({'error': 'postcode and category are required'}), 400

#         # Call the Api function
#         df1, df2 = Api([postcode], category)

#         # Convert DataFrames to JSON
#         df1_json = df1.to_dict(orient='records')
#         df2_json = df2.to_dict(orient='records')

#         # Return the combined JSON response
#         return jsonify({
#             'dataframe1': df1_json,
#             'dataframe2': df2_json
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
from model import Api

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/getlikelihoodscore', methods=['POST'])
def get_likelihood_score():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        if 'postcode' not in data:
            return jsonify({
                'error': 'postcode field is required',
                'status': 'error'
            }), 400
        
        if 'category' not in data:
            return jsonify({
                'error': 'category field is required',
                'status': 'error'
            }), 400
        
        postcode = data['postcode']
        category = data['category']
        
        # Validate data types
        if not isinstance(postcode, list) or not isinstance(category, str):
            return jsonify({
                'error': 'postcode and category must be strings',
                'status': 'error'
            }), 400
        
        print(f'Processing request - Postcode: {postcode}, Category: {category}')
        
        # Call the API function
        result = Api(postcode, category)
        
        # Check if result contains two DataFrames
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            return jsonify({
                'error': 'API function did not return expected format (2 DataFrames)',
                'status': 'error'
            }), 500
        
        df1, df2 = result
        
        # Validate that both results are DataFrames
        if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
            return jsonify({
                'error': 'API function did not return DataFrames',
                'status': 'error'
            }), 500
        
        # Convert DataFrames to JSON
        # Using 'records' orientation to get array of objects
        df1_json = df1.to_dict('records') if not df1.empty else []
        df2_json = df2.to_dict('records') if not df2.empty else []
        
        # Create response
        response = {
            'status': 'success',
            'data': {
                'postcode': postcode,
                'category': category,
                'dataframe1': df1_json,
                'dataframe2': df2_json,
                'dataframe1_shape': df1.shape,
                'dataframe2_shape': df2.shape
            }
        }
        
        print(f'Successfully processed request for {postcode}')
        return jsonify(response), 200
        
    except ImportError as e:
        return jsonify({
            'error': f'Failed to import model: {str(e)}',
            'status': 'error'
        }), 500
        
    except Exception as e:
        print(f'Error processing request: {str(e)}')
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500


if __name__ == '__main__':
    print('Starting Flask API server...')
    
    # Run the app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode for development
    )