import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './data/cleanData.csv'
data = pd.read_csv(file_path)

# Step 1: Parse the Date column and extract month and year
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Step 2: Map months to Nepali seasons
season_map = {
    12: "BASANTA RITU", 1: "BASANTA RITU",
    2: "GRISHMA RITU", 3: "GRISHMA RITU",
    4: "BARSHA RITU", 5: "BARSHA RITU",
    6: "SARAD RITU", 7: "SARAD RITU",
    8: "Hemanta Ritu", 9: "Hemanta Ritu",
    10: "Shishir Ritu", 11: "Shishir Ritu"
}
data['Season'] = data['Month'].map(season_map)

# Step 3: Group data by Commodity, Season, and Year to calculate average prices
seasonal_data = data.groupby(['Commodity', 'Season'], as_index=False)['Average'].mean()
yearly_seasonal_data = data.groupby(['Commodity', 'Season', 'Year'], as_index=False)['Average'].mean()

# Function to ensure the directory exists
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to sanitize commodity name for filenames
def sanitize_filename(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

# Function to plot seasonal analysis (box and line plots)
def save_commodity_seasonal_analysis(commodity_name, df, save_path):
    # Filter data for the specified commodity
    commodity_data = df[df['Commodity'] == commodity_name]
    
    # Order of seasons
    season_order = ["BASANTA RITU", "GRISHMA RITU", "BARSHA RITU", 
                    "SARAD RITU", "Hemanta Ritu", "Shishir Ritu"]
    
    # Ensure directories exist
    create_directory(f"{save_path}/seasonal_trends_box")
    create_directory(f"{save_path}/seasonal_trends_line")
    
    # Sanitize the commodity name for the file name
    commodity_name_safe = sanitize_filename(commodity_name)

    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=commodity_data, x='Season', y='Average', order=season_order)
    plt.title(f"Box Plot of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Average Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_trends_box/box_plot_{commodity_name_safe}.png")
    plt.close()
    
    # Line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=commodity_data, x='Season', y='Average', marker='o', sort=False)
    plt.title(f"Line Graph of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Average Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_trends_line/line_plot_{commodity_name_safe}.png")
    plt.close()

# Function to plot season-year heatmap
def save_commodity_season_year_analysis(commodity_name, df, save_path):
    # Filter data for the specified commodity
    commodity_data = df[df['Commodity'] == commodity_name]
    
    # Order of seasons
    season_order = ["BASANTA RITU", "GRISHMA RITU", "BARSHA RITU", 
                    "SARAD RITU", "Hemanta Ritu", "Shishir Ritu"]
    
    # Pivot the data for heatmap
    pivot_data = commodity_data.pivot(index='Year', columns='Season', values='Average').reindex(columns=season_order)
    
    # Ensure directories exist
    create_directory(f"{save_path}/seasonal_heat_map")
    
    # Sanitize the commodity name for the file name
    commodity_name_safe = sanitize_filename(commodity_name)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Price'})
    plt.title(f"Season-Year Map of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Year", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_heat_map/heatmap_{commodity_name_safe}.png")
    plt.close()

# Example usage: Save plots for all commodities
save_path = './visualizationFig'  # Path to save the plots

# Loop through all unique commodities in the dataset and save the plots for each one
for commodity_name in data['Commodity'].unique():
    save_commodity_seasonal_analysis(commodity_name, data, save_path)
    save_commodity_season_year_analysis(commodity_name, yearly_seasonal_data, save_path)
