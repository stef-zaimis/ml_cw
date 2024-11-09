import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Code Task 1: Import the data
X, y = fetch_covtype(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

#print("Subset X shape:", X_train.shape)
#print("Subset y shape:", y_train.shape)

# Code Task 2: K-means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train) # I scale here to compensate for differing feature ranges, this might improve performance
kmeans = KMeans(n_clusters=7, init='k-means++', random_state=42).fit(X_scaled)

print("K-means cluster centers:\n", kmeans.cluster_centers_)
print("K-means labels for the subset:\n", kmeans.labels_)

