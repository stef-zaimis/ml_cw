import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_covtype(return_X_y=True, as_frame=True)
X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, stratify=y)

# Define hyperparameter grid
init_params_options = ['kmeans', 'random']
max_iter_options = [100, 200]
tol_options = [1e-3, 1e-4]
n_components = 7  # We use 7 clusters as per the task

# Dictionary to store results
results = []

# Loop over hyperparameter combinations
for init in init_params_options:
    for max_iter in max_iter_options:
        for tol in tol_options:
            # Initialize GMM model
            gmm = GaussianMixture(
                n_components=n_components, 
                init_params=init,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            )
            
            # Fit model and predict clusters
            gmm.fit(X_train)
            y_pred = gmm.predict(X_train)

            # Evaluate using Adjusted Rand Index
            ari_score = adjusted_rand_score(y_train, y_pred)
            log_likelihood = gmm.score(X_train)

            # Store results
            results.append({
                'init_params': init,
                'max_iter': max_iter,
                'tol': tol,
                'ARI': ari_score,
                'Log-Likelihood': log_likelihood
            })

# Display the results sorted by ARI and log-likelihood
sorted_results = sorted(results, key=lambda x: (x['ARI'], x['Log-Likelihood']), reverse=True)
for result in sorted_results:
    print(result)

