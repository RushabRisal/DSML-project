import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the dataset
data = pd.read_csv("data/cleanData.csv")
# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Get the list of unique commodities
commodities = data['Commodity'].unique()

# Number of commodities to display per graph
commodities_per_graph = 1
num_graphs = math.ceil(len(commodities) / commodities_per_graph)

# Loop through and create graphs for each set of commodities
for i in range(num_graphs):
    # Select the subset of commodities for this graph
    subset_commodities = commodities[i * commodities_per_graph: (i + 1) * commodities_per_graph]
    
    # Filter data for the subset of commodities
    subset_data = data[data['Commodity'].isin(subset_commodities)]
    
    # Plot daily prices
    plt.figure(figsize=(16, 10))
    for commodity, group in subset_data.groupby('Commodity'):
        plt.plot(group['Date'], group['Average'], label=commodity, alpha=0.7, linewidth=1.5)
    
    # Add title, labels, and legend
    plt.title(f'Daily Prices for Commodities ({i * commodities_per_graph + 1} to {(i + 1) * commodities_per_graph})', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Price', fontsize=14)
    plt.legend(title='Commodity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
