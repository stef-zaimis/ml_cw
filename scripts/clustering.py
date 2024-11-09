import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

# Import the data
X, y = fetch_covtype(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

print("Subset X shape:", X_train.shape)
print("Subset y shape:", y_train.shape)
