from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [7,7],[5,9],[6,9],[7,9],[5,11],[5,3],[6,1],[6,2],
    [7,1],[7,2],[7,3],[8,4],[8,6],[9,3],[9,4],[9,5],
    [10,4],[10,5],[10,6],[9,7]
])

# K-means for 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42)
kmeans_2.fit(data)

# Plotting the data points and clusters
def plot_kmeans(data, kmeans, num_clusters, ax):
    ax.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
    ax.set_title(f'K-means with {num_clusters} Clusters')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Create a figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# K-means for 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42)
kmeans_2.fit(data)
plot_kmeans(data, kmeans_2, 2, axs[0])

# K-means for 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(data)
plot_kmeans(data, kmeans_3, 3, axs[1])

# K-means for 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42)
kmeans_4.fit(data)
plot_kmeans(data, kmeans_4, 4, axs[2])

# Show the plot
plt.tight_layout()
plt.show()