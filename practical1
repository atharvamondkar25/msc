import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))
print(data)

plt.figure(figsize=(5, 4))
plt.title("Original Data Set")
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

kmeans2 = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans2.fit(data)
print("Labels (k=2):", kmeans2.labels_)
print("Centers (k=2):\n", kmeans2.cluster_centers_)

kmeans3 = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans3.fit(data)
print("Labels (k=3):", kmeans3.labels_)
print("Centers (k=3):\n", kmeans3.cluster_centers_)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("Original Data Set")
plt.scatter(x, y, color="blue")

plt.subplot(1, 3, 2)
plt.title("K-Means (k=2)")
plt.scatter(x, y, c=kmeans2.labels_, cmap="rainbow")
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],
            color="black", marker="X", s=100, label="Centers")
plt.legend()

plt.subplot(1, 3, 3)
plt.title("K-Means (k=3)")
plt.scatter(x, y, c=kmeans3.labels_, cmap="rainbow")
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1],
            color="black", marker="X", s=100, label="Centers")
plt.legend()
plt.show()
