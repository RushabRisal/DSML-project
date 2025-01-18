import pandas as pd
from prophet import Prophet
import joblib
from multiprocessing import Pool, cpu_count

data = pd.read_csv('./data/cleanData.csv')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data = (
    data.groupby(['Commodity', 'Date'])
    .agg({'Average': 'mean'})
    .reset_index()
    .rename(columns={'Date': 'ds', 'Average': 'y'})
)
def train_model(commodity_group):
    commodity_name, group_data = commodity_group
    model = Prophet()
    model.fit(group_data)
    model_filename = f"./models/time_prophet_model.joblib"
    joblib.dump(model, model_filename)


if __name__ == "__main__":
    grouped_data = data.groupby('Commodity')
    tasks = [(commodity, group) for commodity, group in grouped_data]
    with Pool(cpu_count() - 1) as pool:  
        results = pool.map(train_model, tasks)
