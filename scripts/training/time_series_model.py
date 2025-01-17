import pandas as pd
from prophet import Prophet
from joblib import dump

# Load and preprocess your data
data = pd.read_csv('./data/cleanData.csv')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = data.groupby(['Commodity', 'Date']).agg({'Average': 'mean'}).reset_index()
data = data[['Commodity', 'Date', 'Average']].rename(columns={'Date': 'ds', 'Average': 'y'})
commodity_groups = data.groupby('Commodity')
def train_and_save_model(group, commodity_name):
    model = Prophet()
    model.fit(group)
    dump(model, f'./models/time_prophet_model.pkl')
for commodity, group in commodity_groups:
    train_and_save_model(group, commodity)
