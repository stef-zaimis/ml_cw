import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_covtype
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Code Task 1: Import the data
X, y = fetch_covtype(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, stratify=y)

# print("Subset X shape:", X_train.shape)
# print("Subset y shape:", y_train.shape)

# Code Task 2: K-means clustering
K = 7
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    X_train
)  # I scale here to compensate for differing feature ranges, this might improve performance
kmeans = KMeans(n_clusters=K, init="k-means++", random_state=42).fit(X_scaled)

print("K-means cluster centers:\n", kmeans.cluster_centers_)
print("K-means labels for the subset:\n", kmeans.labels_)

# Code Task 3: Gaussian mixtures
gmm = GaussianMixture(
    n_components=K, max_iter=100, init_params="kmeans", tol=1e-3, random_state=42
).fit(X_scaled)

print("Gaussian Mixture Model means:\n", gmm.means_)
print("Gaussian Mixture Model covariances:\n", gmm.covariances_)

# Code Task 4: Random baseline
random_labels = np.random.randint(0, K, size=len(X_train))

# Code Task 5: Error counting

kmeans_labels = kmeans.labels_
gmm_labels = gmm.predict(X_scaled)

# print("K-means clustering labels:\n", kmeans_labels, " of length: ", len(kmeans_labels))
# print("GMM clustering labels:\n", gmm_labels, " of length: ", len(gmm_labels))
# print("Random baseline clustering labels:\n", random_labels, " of length: ", len(random_labels))


# Function to get unique pairs in a list. This is a modified version of user7864386's solution here: https://stackoverflow.com/questions/70413515/get-all-unique-pairs-in-a-list-including-duplicates-in-python
def get_pairs(l):
    out = []
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            out.append((l[i], l[j]))
    return out


def count_errors(true_labels, cluster_labels):
    error_count = 0
    total_pairs = 0
    label_dict = {}

    for idx, label in enumerate(true_labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(idx)

    for indices in label_dict.values():
        n = len(indices)
        num_pairs = n * (n - 1) // 2
        total_pairs += num_pairs

        for i, j in get_pairs(indices):
            if (
                cluster_labels[i] != cluster_labels[j]
                and true_labels[i] == true_labels[j]
            ):
                error_count += 1

    print(f"Total pairs: {total_pairs}")
    return error_count


kmeans_errors = count_errors(y_train.values, kmeans_labels)
gmm_errors = count_errors(y_train.values, gmm_labels)
random_errors = count_errors(y_train.values, random_labels)

print(f"Errors in K-means clustering: {kmeans_errors}")
print(f"Errors in GMM clustering: {gmm_errors}")
print(f"Errors in random baseline clustering: {random_errors}")
