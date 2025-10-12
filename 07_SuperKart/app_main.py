import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "deployment_files", "superkart_model_v1_0.joblib")
#model = joblib.load("deployment\superkart_model_v1_0.joblib")
print("Loading model from:", model_path)
model = joblib.load(model_path)

# (Optional) Load any encoders or scalers if used
# encoder = joblib.load('encoder.joblib')
# scaler = joblib.load('scaler.joblib')

st.title("SuperKart Total Sales Prediction")

# Define the input fields (replace with your actual feature names)
product_weight = st.number_input("Product Weight", min_value=0.0)
sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar", "reg"])
allocated_area = st.number_input("Allocated Area", min_value=0.0)
product_type = st.selectbox("Product Type", sorted([
    "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Snack Foods",
    "Meat", "Household", "Fruits and Vegetables", "Breads", "Hard Drinks", "Soft Drinks",
    "Breakfast", "Starchy Foods", "Seafood", "Others"
]))
product_mrp = st.number_input("Product MRP", min_value=0.0)
store_id = st.text_input("Store ID")
store_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2100)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
])

# Collect all inputs into a DataFrame
input_dict = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": sugar_content,
    "Product_Allocated_Area": allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type
}

print("Input dict______", input_dict)

input_df = pd.DataFrame([input_dict])

print("")
print("Input df___________", input_df)

# (Optional) Apply any preprocessing here
# input_df = encoder.transform(input_df)
# input_df = scaler.transform(input_df)

if st.button("Predict Total Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted SuperKart Total Sales: {prediction:.2f}")