import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# Load dataset
st.title("Vegetable Market Analysis and Prediction")
st.text("Visualization of the data")

# Define the specific commodities to display
selected_commodities = [
    'Amla', 'Apple(Fuji)', 'Apple(Jholey)', 'Arum', 'Asparagus',
    'Avocado', 'Bakula', 'Bamboo Shoot', 'Banana', 'Barela'
]

# Load dataset and filter for selected commodities
dataframe = pd.read_csv('./data/cleanData.csv')
filtered_df = dataframe[dataframe['Commodity'].isin(selected_commodities)]

st.dataframe(filtered_df.head(100))

description = filtered_df.describe()
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


def plot_correlation_heatmap():
    # Filter the data to include only the top 10 commodities by average price
    top_commodities = dataframe.groupby('Commodity')['y'].mean().nlargest(10).index
    filtered_data = dataframe[dataframe['Commodity'].isin(top_commodities)]
    
    pivot_table = filtered_data.pivot_table(values='y', index='ds', columns='Commodity')
    correlation_matrix = pivot_table.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))  # Increase the figure size
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=.5)
    ax.set_title('Correlation Heatmap of Top 10 Commodities')
    st.pyplot(fig)

def plot_moving_average(commodity_name, window=30):
    commodity_data = dataframe[dataframe['Commodity'] == commodity_name]
    commodity_data['Moving_Avg'] = commodity_data['y'].rolling(window=window).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(commodity_data)), commodity_data['y'], label='Original')
    ax.plot(range(len(commodity_data)), commodity_data['Moving_Avg'], label='Moving Average', color='red')
    ax.set_title(f'Moving Average (window={window}) for {commodity_name}')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks([])
    plt.tight_layout()
    st.pyplot(fig)

def calculate_seasonal_averages(commodity_name):
    # Define the seasons
    seasons = {
        'Basanta': [3, 4],      # Chaitra & Baisakh (March-April)
        'Grishma': [5, 6],      # Jestha & Asar (May-June)
        'Barsha': [7, 8],       # Shrawan & Bhadra (July-August)
        'Sharad': [9, 10],      # Ashoj & Kartik (September-October)
        'Hemanta': [11, 12],    # Mangsir & Poush (November-December)
        'Shishir': [1, 2]       # Magh & Falgun (January-February)
    }
    
    # Filter the data for the selected commodity
    commodity_data = dataframe[dataframe['Commodity'] == commodity_name].copy()

    # Add a 'Season' column to the dataframe using .loc
    commodity_data.loc[:, 'Season'] = commodity_data['ds'].apply(lambda x: next(season for season, months in seasons.items() if pd.to_datetime(x).month in months))

    # Calculate seasonal averages
    seasonal_averages = commodity_data.groupby(['Season'])['y'].mean().reset_index()

    # Ensure all seasons are included
    all_seasons = pd.DataFrame(list(seasons.keys()), columns=['Season'])
    seasonal_averages = all_seasons.merge(seasonal_averages, on='Season', how='left').fillna(np.nan)

    # Create a styled dataframe
    def highlight_minmax(x):
        if x == seasonal_averages['y'].max():
            return 'background-color: #ffcdd2'  # light red
        elif x == seasonal_averages['y'].min():
            return 'background-color: #c8e6c9'  # light green
        else:
            return ''

    styled_df = seasonal_averages.style.apply(lambda x: [highlight_minmax(v) for v in x], subset=['y'])

    st.write(f"Seasonal Averages for {commodity_name}:")
    # Set wider width for the table
    st.dataframe(styled_df, width=800)

def calculate_yearly_trends(commodity_name):
    # Define the seasons
    seasons = {
        'Basanta': [3, 4],      # Chaitra & Baisakh (March-April)
        'Grishma': [5, 6],      # Jestha & Asar (May-June)
        'Barsha': [7, 8],       # Shrawan & Bhadra (July-August)
        'Sharad': [9, 10],      # Ashoj & Kartik (September-October)
        'Hemanta': [11, 12],    # Mangsir & Poush (November-December)
        'Shishir': [1, 2]       # Magh & Falgun (January-February)
    }
    
    # Filter the data for the selected commodity
    commodity_data = dataframe[dataframe['Commodity'] == commodity_name].copy()
    
    # Add 'Year' and 'Season' columns to the dataframe
    commodity_data.loc[:, 'Year'] = pd.to_datetime(commodity_data['ds']).dt.year
    commodity_data.loc[:, 'Season'] = commodity_data['ds'].apply(lambda x: next(season for season, months in seasons.items() if pd.to_datetime(x).month in months))
    
    # Calculate yearly trends
    yearly_trends = commodity_data.groupby(['Season', 'Year'])['y'].mean().reset_index()

    # Ensure all seasons are included
    all_seasons = pd.DataFrame(list(seasons.keys()), columns=['Season'])
    yearly_trends = all_seasons.merge(yearly_trends, on='Season', how='left').fillna(np.nan)
    
    # Create a styled dataframe
    def highlight_minmax(x):
        if x == yearly_trends['y'].max():
            return 'background-color: #ffcdd2'  # light red
        elif x == yearly_trends['y'].min():
            return 'background-color: #c8e6c9'  # light green
        else:
            return ''

    styled_df = yearly_trends.style.apply(lambda x: [highlight_minmax(v) for v in x], subset=['y'])

    st.write(f"Yearly Trends for {commodity_name}:")
    # Set wider width for the table
    st.dataframe(styled_df, width=800)
    
    # Plot the yearly trends
    fig, ax = plt.subplots(figsize=(10, 6))
    for season in yearly_trends['Season'].unique():
        season_data = yearly_trends[yearly_trends['Season'] == season]
        ax.plot(season_data['Year'], season_data['y'], marker='o', label=season)
    
    ax.set_title(f"Yearly Trends for {commodity_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Price")
    ax.legend(title="Season")
    plt.xticks(yearly_trends['Year'].unique())
    plt.tight_layout()
    st.pyplot(fig)

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

if st.button("Correlation Heatmap"):
    plot_correlation_heatmap()

if st.button("Moving Average"):
    plot_moving_average(commodity_name)

if st.button("Calculate Seasonal Averages"):
    calculate_seasonal_averages(commodity_name)

if st.button("Calculate Yearly Trends"):
    calculate_yearly_trends(commodity_name)

kmeans = joblib.load("./models/kmeans_model.joblib")
scaler = joblib.load("./models/scaler.joblib")
df_grouped = pd.read_csv("./data/clustered_commodities.csv")

st.title("Commodity Cluster Predictor")

# User inputs for commodity name and new average price
commodity_input = st.text_input("Enter Commodity Name")
user_avg_price = st.number_input("Enter Commodity Average Price", min_value=0.0, format="%.2f")

if st.button("Predict Cluster"):
    if commodity_input in df_grouped["Commodity"].values:
        # Retrieve the stored min and max values for this commodity
        record = df_grouped[df_grouped["Commodity"] == commodity_input].iloc[0]
        original_min = record["Minimum"]
        original_max = record["Maximum"]
        input_features = np.array([[original_min, original_max, user_avg_price]])
        scaled_input = scaler.transform(input_features)
        
        # Predict the cluster using the loaded K-means model
        predicted_cluster = kmeans.predict(scaled_input)[0]
        st.success(f"The commodity '{commodity_input}' with an average price of {user_avg_price} belongs to Cluster {predicted_cluster}.")
        training_features = df_grouped[["Minimum", "Maximum", "Average"]].values
        scaled_training = scaler.transform(training_features)
        
        # Apply PCA to reduce the dimensions to 2 for plotting
        pca = PCA(n_components=2)
        reduced_training = pca.fit_transform(scaled_training)
        
        # Transform the new commodity's scaled input with the same PCA model
        reduced_new_point = pca.transform(scaled_input)
        
        # Create the scatter plot for training data
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            reduced_training[:, 0], 
            reduced_training[:, 1], 
            c=df_grouped['Cluster'], 
            cmap='viridis', 
            alpha=0.6, 
            label="Existing Commodities"
        )
        ax.scatter(
            reduced_new_point[0, 0], 
            reduced_new_point[0, 1], 
            c='red', 
            marker='X', 
            s=200, 
            label="New Commodity"
        )
        
        ax.set_xlabel("Overall Level")
        ax.set_ylabel("Price Spread")
        ax.set_title("Commodity Clusters with New Commodity Highlighted")
        ax.legend()
        
        st.pyplot(fig)
    else:
        st.error("Commodity not found in dataset. Please check the name and try again.")