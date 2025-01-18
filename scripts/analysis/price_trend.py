import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loading the dataset
df = pd.read_csv("./data/cleanData.csv")

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Group by date and commodity
trend_data = df[['Commodity', 'Date', 'Average']]
daily_trend = trend_data.groupby(['Date', 'Commodity'])['Average'].mean().reset_index()

# Get unique commodities
commodities = daily_trend['Commodity'].unique()

# Split commodities into groups of 10
commodity_groups = [commodities[i:i + 1] for i in range(0, len(commodities), 1)]

# Plot each group separately
for i, group in enumerate(commodity_groups):
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=daily_trend[daily_trend['Commodity'].isin(group)], x='Date', y='Average', hue='Commodity')
    plt.title(f"Daily Price Trend for {group[0]}")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend(title="Commodity")
    plt.savefig(f"./visualizationFig/price_trends/{group[0]}.png")
    plt.close()