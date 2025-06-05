import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

sns.set(style="whitegrid")

blob_centers = [[1.5, 1.5], [-1.5, -1.0]]
std_devs = [0.4, 0.6]
data, _ = make_blobs(n_samples=250, centers=blob_centers, cluster_std=std_devs, random_state=7)

random_noise = np.random.uniform(low=-5, high=5, size=(40, 2))
data_all = np.concatenate([data, random_noise], axis=0)

db = DBSCAN(eps=0.5, min_samples=6)
cluster_labels = db.fit_predict(data_all)

plt.figure(figsize=(8, 6))
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
palette = sns.color_palette("hsv", n_colors=n_clusters)

for label in set(cluster_labels):
    mask = (cluster_labels == label)
    if label == -1:
        # Black for noise
        color = (0, 0, 0, 1)
    else:
        color = palette[label]
    plt.scatter(data_all[mask, 0], data_all[mask, 1], s=40, color=color, edgecolors='k', label=f"Cluster {label}" if label != -1 else "Noise")

plt.title("DBSCAN Result with Noise")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
