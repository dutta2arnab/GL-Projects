# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_forecast_api = Flask("Superkart Sales Forecast")

# Load the trained machine learning model
model = joblib.load("deployment_files/superkart_model_v1_0.joblib")

# Define a route for the home page (GET request)
@superkart_sales_forecast_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Superkart Sales Forecast API!"

# Define an endpoint for single property prediction (POST request)
@superkart_sales_forecast_api.post('/v1/predict_sales')
def predict_sales_forecast():
    """
    This function handles POST requests to the '/v1/rental' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    superkart_data = request.get_json()

    print("Superkart Data_____",superkart_data)
    print("Product Weight__",superkart_data['product_weight'])

    # Extract relevant features from the JSON data
    sample = {
        "Product_Weight": superkart_data['product_weight'],
        "Product_Sugar_Content": superkart_data['product_sugar_content'],
        "Product_Allocated_Area": superkart_data['product_allocated_area'],
        "Product_Type": superkart_data['product_type'],
        "Product_MRP": superkart_data['product_mrp'],
        "Store_Id": superkart_data['store_id'],
        "Store_Establishment_Year": superkart_data['store_establishment_year'],
        "Store_Size": superkart_data['store_size'],
        "Store_Location_City_Type": superkart_data['store_location_city_type'],
        "Store_Type" : superkart_data['store_type']
    }

    print("Sample Data_____",sample)

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    print("Input_data__________",input_data)
    print("")

    print("Model Features Names______",model.feature_names_in_)
    

    # Make prediction (get log_price)
    predicted_sales_price = model.predict(input_data)[0]
    print("Predicted Price is _________",predicted_sales_price)

    # Calculate actual price
    #predicted_price = np.exp(predicted_log_price)

    # Convert predicted_price to Python float
    predicted_price = round(float(predicted_sales_price), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted Price (in dollars)': predicted_price})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_forecast_api.post('/v1/predict_sales_batch')
def predict_sales_forecast_batch():
    """
    This function handles POST requests to the '/v1/predict_sales_batch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_price = model.predict(input_data).tolist()

    # Calculate actual prices
    #predicted_prices = [round(float(np.exp(log_price)), 2) for log_price in predicted_log_prices]

    # Create a dictionary of predictions with property IDs as keys
    product_ids = input_data['product_id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(product_ids, predicted_price))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_forecast_api.run(debug=True)
