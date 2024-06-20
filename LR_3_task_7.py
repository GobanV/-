import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

input_file = 'data_clustering.txt'
data = np.loadtxt(input_file, delimiter=',')

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Кластеризація методом k-means')
plt.show()

with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans, model_file)
