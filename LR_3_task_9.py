import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

X = np.loadtxt('data_clustering.txt', delimiter=',')

mean_shift = MeanShift()
mean_shift.fit(X)
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Кластеризація методом зсуву середнього')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

print("Координати центрів кластерів:")
print(cluster_centers)
