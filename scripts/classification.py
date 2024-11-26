import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Import covtype
X, y = fetch_covtype(return_X_y=True, as_frame=True)

# Code Task 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Code Task 7
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

log_reg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_logreg = log_reg.predict(X_test_scaled)
accuracy_logreg = np.count_nonzero(y_pred_logreg==np.int64(y_test))/y_test.shape[0]

print(f"Log Reg Test Set Accurracy: {accuracy_logreg}")

# Code Task 8
tree_clf = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_clf_cv_scores = cross_val_score(tree_clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Decision Tree Cross-Validation Accuracy: {tree_clf_cv_scores.mean():.4f}")


tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
accuracy_tree = np.count_nonzero(y_pred_tree==np.int64(y_test))/y_test.shape[0]

print(f"Tree Test Set Accuracy: {accuracy_tree}")

# Code Task 9
# I will implement both bagging and boosting for now to see which one performs better and which one is more suitable (probaby random forest)

# RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt', random_state=42)
rf_cv_scores = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean():.4f}")

random_forest.fit(X_train_scaled, y_train)

y_pred_rf = random_forest.predict(X_test_scaled)
rf_accuracy = np.count_nonzero(y_pred_rf==np.int64(y_test))/y_test.shape[0]

print(f"Bagging Test SEt Accuracy: {rf_accuracy}")

# GRADIENT BOOSTING
#gradient_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
#gradient_boost.fit(X_train, y_train)

#y_pred_boost = gradient_boost.predict(X_test)
#boost_accuracy = np.count_nonzero(y_pred_boost==np.int64(y_test))/y_test.shape[0]

#print(f"Boosting Test SEt Accuracy: {boost_accuracy}")
