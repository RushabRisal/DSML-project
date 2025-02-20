import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load dataset
st.title("Vegetable Market Analysis and Prediction")
st.text("Visualization of the data")

dataframe = pd.read_csv('./data/cleanData.csv')
st.dataframe(dataframe.head(100))

description = dataframe.describe()
st.write(description)

# Model directory
model_dir = "./models"
dataframe = (
    dataframe.groupby(['Commodity', 'Date'])
    .agg({'Average': 'mean'})
    .reset_index()
    .rename(columns={'Date': 'ds', 'Average': 'y'})
)
selected_commodities = dataframe['Commodity'].unique()[:10]
# User inputs
commodity_name = st.selectbox("Select a commodity", selected_commodities)
target_date = st.date_input("Select a date for prediction")


def predict_price(commodity_name, target_date):
    model_path = os.path.join(model_dir, f"prophet_{commodity_name}.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model for {commodity_name} not found.")
        return None
    
    model = joblib.load(model_path)
    new_data = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
    forecast = model.predict(new_data)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return forecast

def classify_price_fluctuation(commodity_name, target_date):
    model_path = os.path.join(model_dir, "price_fluctuation_model.joblib")
    if not os.path.exists(model_path):
        st.error("Price fluctuation model not found.")
        return None
    
    model = joblib.load(model_path)
    # Use the same feature names as during training
    new_data = pd.DataFrame({'Average': [dataframe[dataframe['Commodity'] == commodity_name]['y'].mean()]})
    prediction = model.predict(new_data)
    return prediction[0]

if st.button("Predict Price"):
    forecast = predict_price(commodity_name, target_date)
    if forecast is not None:
        st.write(f"Prediction for {commodity_name} on {target_date}:")
        st.write(forecast)
        
        # Visualization
        model = joblib.load(os.path.join(model_dir, f"prophet_{commodity_name}.joblib"))
        future = model.make_future_dataframe(periods=500, freq='D')
        forecast_full = model.predict(future)
        fig, ax = plt.subplots()
        ax.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecast')
        ax.fill_between(forecast_full['ds'], forecast_full['yhat_lower'], forecast_full['yhat_upper'], color='gray', alpha=0.2)
        ax.set_title(f"Forecast for {commodity_name}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted Price (yhat)")
        st.pyplot(fig)

if st.button("Classify Price Fluctuation"):
    fluctuation = classify_price_fluctuation(commodity_name, target_date)
    if fluctuation is not None:
        fluctuation_text = "Expected to Increase" if fluctuation == 1 else "Expected to Decrease"
        st.write(f"Price fluctuation prediction for {commodity_name} on {target_date}: {fluctuation_text}")
        
        # Get historical data for the selected commodity
        commodity_data = dataframe[dataframe['Commodity'] == commodity_name]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pd.to_datetime(commodity_data['ds']), commodity_data['y'], label='Historical Prices')
        ax.set_title(f"Historical Price Trends for {commodity_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        
        # Add a horizontal line for the mean price
        mean_price = commodity_data['y'].mean()
        ax.axhline(y=mean_price, color='r', linestyle='--', label='Mean Price')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)