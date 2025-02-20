import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from joblib import dump
# Step 1: Load the Dataset
file_path = "./data/cleanData.csv"
df = pd.read_csv(file_path)

# Step 2: Aggregate Data by Commodity (Mean of Prices)
df_grouped = df.groupby("Commodity")[["Minimum", "Maximum", "Average"]].mean().reset_index()

# Step 3: Standardize the Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_grouped[["Minimum", "Maximum", "Average"]])

# Step 4: Determine Optimal K using Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot and Save the Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

vis_dir = "./visualizationFig"
os.makedirs(vis_dir, exist_ok=True)
plt.savefig(os.path.join(vis_dir, f"elbow_method.png"))
plt.show()

# Step 5: Train the K-means Model
optimal_k = 4  # Choose based on the elbow method plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_grouped['Cluster'] = kmeans.fit_predict(scaled_data)

# Save Clustered Data to CSV
df_grouped.to_csv("./data/clustered_commodities.csv", index=False)
dump(kmeans, "./models/kmeans_model.joblib")
dump(scaler, "./models/scaler.joblib")
# Step 6: Visualize Clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df_grouped['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Commodity Clusters')
plt.colorbar(label='Cluster')
# plt.savefig("/visualizaitonFig/commodity_clusters.png")
vis_dir = "./visualizationFig"
os.makedirs(vis_dir, exist_ok=True)
plt.savefig(os.path.join(vis_dir, f"commodity_clusters.png"))
plt.show()

# Display Sample Clustered Commodities
print(df_grouped[['Commodity', 'Cluster']].head())
