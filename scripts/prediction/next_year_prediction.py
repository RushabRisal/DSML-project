import joblib
import pandas as pd
from prophet import Prophet


def load_model(commodity_name):
    try:
        return joblib.load(f'./models/time_prophet_model.pkl')
    except Exception as e:
        print(f"Error loading model for {commodity_name}: {e}")
        return None
def predict_for_date(model, date, commodity_name):
    try:
        future_data = pd.DataFrame({'ds': [pd.to_datetime(date)]})
        forecast = model.predict(future_data)
        return {
            'Commodity': commodity_name,
            'Date': date,
            'yhat': forecast['yhat'].iloc[0],
            'yhat_lower': forecast['yhat_lower'].iloc[0],
            'yhat_upper': forecast['yhat_upper'].iloc[0],
        }
    except Exception as e:
        print(f"Error during prediction for {commodity_name}: {e}")
        return None


def get_predictions_by_commodity(date, commodity_list):
    predictions = []
    
    for commodity_name in commodity_list:
        model = load_model(commodity_name)
        if model is not None:
            result = predict_for_date(model, date, commodity_name)
            if result is not None:
                predictions.append(result)
    
    if predictions:
        return pd.DataFrame(predictions)
    else:
        return "No valid predictions available."

commodity_list = ['Tomato', 'Potato', 'Onion', 'Carrot']
commodity_name='Onion'
date = '2030-02-01'  
detailed_predictions = predict_for_date(load_model(commodity_name),date, commodity_name)
print(detailed_predictions)
