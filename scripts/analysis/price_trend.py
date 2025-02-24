import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("./data/cleanData.csv")

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Group by date and commodity
trend_data = df[['Commodity', 'Date', 'Average']]
daily_trend = trend_data.groupby(['Date', 'Commodity'])['Average'].mean().reset_index()

commodities = daily_trend['Commodity'].unique()

# Plot for each commodity
for commodity in commodities:
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=daily_trend[daily_trend['Commodity'] == commodity], x='Date', y='Average')
    plt.title(f"Daily Price Trend for {commodity}")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.savefig(f"./visualizationFig/price_trends_line/{commodity}.png")
    plt.close()
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=daily_trend[daily_trend['Commodity'] == commodity], x='Commodity', y='Average')
    plt.title(f"Price Distribution for {commodity}")
    plt.xlabel("Commodity")
    plt.ylabel("Average Price")
    plt.savefig(f"./visualizationFig/price_trends_box/{commodity}_box_plot.png")
    plt.close()