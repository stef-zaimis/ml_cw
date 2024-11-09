import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Code Task 1: Import the data
X, y = fetch_covtype(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, stratify=y)

#print("Subset X shape:", X_train.shape)
#print("Subset y shape:", y_train.shape)

# Code Task 2: K-means clustering
K = 7
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train) # I scale here to compensate for differing feature ranges, this might improve performance
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42).fit(X_scaled)

#print("K-means cluster centers:\n", kmeans.cluster_centers_)
#print("K-means labels for the subset:\n", kmeans.labels_)

# Code Task 2: Gaussian mixtures
gmm = GaussianMixture(n_components=K, max_iter=100, init_params='kmeans', tol=1e-3, random_state=42).fit(X_scaled)

#print("Gaussian Mixture Model means:\n", gmm.means_)
#print("Gaussian Mixture Model covariances:\n", gmm.covariances_)

