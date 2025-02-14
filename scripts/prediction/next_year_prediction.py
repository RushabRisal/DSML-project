import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define model directory
model_dir = "./models"

# Function to load model and predict for a given commodity and date
def predict_price(commodity_name, target_date):
    model_path = os.path.join(model_dir, f"prophet_{commodity_name}.joblib")
    
    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {commodity_name} not found.")
    
    model = joblib.load(model_path)
    
    # Create a DataFrame for the target date
    new_data = pd.DataFrame({'ds': pd.to_datetime([target_date])})
    
    # Make a prediction for the target date
    forecast = model.predict(new_data)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return forecast

# Example usage
if __name__ == "__main__":
    commodity_name = "Amla"  # Change to the desired commodity
    target_date = "2030-01-12"  # Change to the desired date
    
    try:
        forecast = predict_price(commodity_name, target_date)
        print(forecast)
        
        # Visualize the forecasted trend
        model = joblib.load(os.path.join(model_dir, f"prophet_{commodity_name}.joblib"))
        future = model.make_future_dataframe(periods=500, freq='D')
        forecast_full = model.predict(future)
        fig = model.plot(forecast_full)
        
        plt.title(f'Forecast for {commodity_name}')
        plt.xlabel('Year')
        plt.ylabel('Predicted Price (yhat)')
        
        # Save visualization
        vis_dir = "./visualizationFig"
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f"{commodity_name}_forecast.png"))
        plt.show()
        
    except FileNotFoundError as e:
        print(e)
