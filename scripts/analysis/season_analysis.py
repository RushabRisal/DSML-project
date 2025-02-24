import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = './data/cleanData.csv'
data = pd.read_csv(file_path)

# Parse Date column and extract month/year
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Map months to Nepali seasons
season_map = {
    3: "Basanta", 4: "Basanta",
    5: "Grishma", 6: "Grishma", 
    7: "Barsha", 8: "Barsha",
    9: "Sharad", 10: "Sharad",
    11: "Hemanta", 12: "Hemanta",
    1: "Shishir", 2: "Shishir"
}
data['Season'] = data['Month'].map(season_map)

seasonal_data = data.groupby(['Commodity', 'Season'], as_index=False)['Average'].mean()
yearly_seasonal_data = data.groupby(['Commodity', 'Season', 'Year'], as_index=False)['Average'].mean()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sanitize_filename(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")

def save_commodity_seasonal_analysis(commodity_name, df, save_path):
    commodity_data = df[df['Commodity'] == commodity_name]
    season_order = ["Basanta", "Grishma", "Barsha", "Sharad", "Hemanta", "Shishir"]
    
    create_directory(f"{save_path}/seasonal_trends_box")
    create_directory(f"{save_path}/seasonal_trends_line")
    
    commodity_name_safe = sanitize_filename(commodity_name)
    present_seasons = commodity_data['Season'].unique()
    filtered_season_order = [season for season in season_order if season in present_seasons]

    # Generate box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=commodity_data, x='Season', y='Average', order=filtered_season_order)
    plt.title(f"Box Plot of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Average Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_trends_box/box_plot_{commodity_name_safe}.png")
    plt.close()
    
    # Generate line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=commodity_data, x='Season', y='Average', marker='o', sort=False)
    plt.title(f"Line Graph of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Average Price", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_trends_line/line_plot_{commodity_name_safe}.png")
    plt.close()

def save_commodity_season_year_analysis(commodity_name, df, save_path):
    commodity_data = df[df['Commodity'] == commodity_name]
    season_order = ["Basanta", "Grishma", "Barsha", "Sharad", "Hemanta", "Shishir"]
    
    pivot_data = commodity_data.pivot(index='Year', columns='Season', values='Average').reindex(columns=season_order)
    create_directory(f"{save_path}/seasonal_heat_map")
    commodity_name_safe = sanitize_filename(commodity_name)
    
    # Generate heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Price'})
    plt.title(f"Season-Year Map of Average Prices for {commodity_name}", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Year", fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f"{save_path}/seasonal_heat_map/heatmap_{commodity_name_safe}.png")
    plt.close()

save_path = './visualizationFig/Seasonal'

for commodity_name in data['Commodity'].unique():
    save_commodity_seasonal_analysis(commodity_name, data, save_path)
    save_commodity_season_year_analysis(commodity_name, yearly_seasonal_data, save_path)