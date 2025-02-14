import pandas as pd
from prophet import Prophet
import joblib
from multiprocessing import Pool, cpu_count
import os

# Load dataset
data = pd.read_csv('./data/cleanData.csv')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = (
    data.groupby(['Commodity', 'Date'])
    .agg({'Average': 'mean'})
    .reset_index()
    .rename(columns={'Date': 'ds', 'Average': 'y'})
)
selected_commodities = data['Commodity'].unique()[:10]
data = data[data['Commodity'].isin(selected_commodities)]
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

def train_and_save_model(commodity_name, group_data):
    model = Prophet()
    model.fit(group_data)
    model_filename = os.path.join(model_dir, f"prophet_{commodity_name}.joblib")
    joblib.dump(model, model_filename)
    print(f"Model saved for {commodity_name}")

if __name__ == "__main__":
    grouped_data = data.groupby('Commodity')
    tasks = [(commodity, group) for commodity, group in grouped_data]
    with Pool(cpu_count() - 1) as pool:
        pool.starmap(train_and_save_model, tasks)
