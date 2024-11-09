from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess data
X, y = fetch_covtype(return_X_y=True, as_frame=True)
X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, stratify=y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Set number of clusters
K = 7

# Run K-means with k-means++ initialization
kmeans_plus_plus = KMeans(n_clusters=K, init='k-means++', random_state=42).fit(X_scaled)
inertia_plus_plus = kmeans_plus_plus.inertia_
iterations_plus_plus = kmeans_plus_plus.n_iter_

# Run K-means with random initialization
kmeans_random = KMeans(n_clusters=K, init='random', random_state=42).fit(X_scaled)
inertia_random = kmeans_random.inertia_
iterations_random = kmeans_random.n_iter_

# Compare inertia and iterations
print(f"Inertia with k-means++: {inertia_plus_plus}")
print(f"Inertia with random: {inertia_random}")
print(f"Iterations with k-means++: {iterations_plus_plus}")
print(f"Iterations with random: {iterations_random}")

