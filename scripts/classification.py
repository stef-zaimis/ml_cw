import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Import covtype
X, y = fetch_covtype(return_X_y=True, as_frame=True)

# Code Task 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(f"Training set size: {X_train.shape[0]} samples")
#print(f"Test set size: {X_test.shape[0]} samples")

# Code Task 7
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_logreg = log_reg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

#print(f"Log Reg Test Set Accurracy: {accuracy_logreg}")

# Code Task 8
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_scaled, y_train)

y_pred_tree = tree_clf.predict(X_test_scaled)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

print(f"Tree Test Set Accuracy: {accuracy_tree}")
