import joblib
import pandas as pd
import matplotlib.pyplot as plt
# Load a model and make predictions

commodity_name = "Yam"
model_path = f"./models/time_prophet_model.joblib"
model = joblib.load(model_path)
df = pd.read_csv('./data/cleanData.csv')

# Correct the DataFrame creation for the input date
data = pd.DataFrame({'Date': ['2030-01-12']})  
new_data = pd.DataFrame({'ds': pd.to_datetime(data['Date'])})
future = model.make_future_dataframe(periods= 500,freq='D')
# Predict for the given data without extending the future
forecast = model.predict(future)


#visuals of forecast 
fig2= model.plot(forecast)
plt.title('Visualization of the forecast:')
plt.xlabel('Year')
plt.ylabel('yhat')
plt.savefig('./visualizationFig/time_series_data.png')