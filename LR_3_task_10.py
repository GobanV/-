import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target

affinity_propagation = AffinityPropagation(random_state=0)
affinity_propagation.fit(X)
labels = affinity_propagation.labels_
cluster_centers_indices = affinity_propagation.cluster_centers_indices_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Кластеризація з використанням моделі поширення подібності')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

print("Індекси центрів кластерів:")
print(cluster_centers_indices)
