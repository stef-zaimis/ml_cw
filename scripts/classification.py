import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Import covtype
X, y = fetch_covtype(return_X_y=True, as_frame=True)

# Code Task 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
accuracy_tree = np.count_nonzero(y_pred_tree==np.int64(y_test))/y_test.shape[0]

print(f"Tree Test Set Accuracy: {accuracy_tree}")

# Code Task 9
# I will implement both bagging and boosting for now to see which one performs better and which one is more suitable (probaby random forest)

# RANDOM FOREST
num_models = 20
sample_size = 2000
feature_sample_size = None
np.random.seed(0)

all_rf_models = []
for m in range(num_models):
    sample_idx = np.random.choice(X_train.shape[0], sample_size)
    X_train_sample, y_train_sample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]

    rf_model = DecisionTreeClassifier(max_features=feature_sample_size)
    rf_model.fit(X_train_sample, y_train_sample)

    all_rf_models.append(rf_model)

def random_forest_predict(X_test, models):
    votes = np.zeros((X_test.shape[0], len(models)))
    combined_predictions = np.zeros(X_test.shape[0])
    
    for idx, model in enumerate(models):
        votes[:, idx] = model.predict(X_test)
    
    for test_point in range(votes.shape[0]):
        combined_predictions[test_point] = np.bincount(np.int64(votes[test_point])).argmax()
    
    return combined_predictions

rf_predictions = random_forest_predict(X_test, all_rf_models)
rf_accuracy = np.count_nonzero(rf_predictions==np.int64(y_test))/y_test.shape[0]

print(f"Bagging Test Set Accuracy: {rf_accuracy}")

# GRADIENT BOOSTING
num_boost_models = 50
sample_size = 2000
max_depth = 3
np.random.seed(0)

sample_weights = np.ones(X_train.shape[0])/X_train.shape[0]

alphas = []
all_boost_models = []

for m in range(num_boost_models):
    sample_idx = np.random.choice(X_train.shape[0], sample_size)
    X_train_sample, y_train_sample = X_train.loc[sample_idx], y_train.loc[sample_idx]

    boost_model = DecisionTreeClassifier(max_depth=max_depth)
    boost_model.fit(X_train_sample, y_train_sample, sample_weight=sample_weights[sample_idx])
    
    predictions = boost_model.predict(X_train)
    incorrect = (predictions != y_train)
    error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
    
    alpha = 0.5*np.log((1-error)/(error + 1e-10))
    alphas.append(alpha)

    all_boost_models.append(boost_model)
    
    sample_weights *= np.exp(alpha*incorrect)
    sample_weights /= np.sum(sample_weights)

def boosting_predict(X_test, models, alphas):
    votes = np.zeros((X_test.shape[0], len(models)))
    combined_predictions = np.zeros(X_test.shape[0])

    for idx, model in enumerate(models):
        votes[:, idx] = model.predict(X_test)

    for test_point in range(len(votes)):
        combined_predictions[test_point] = np.bincount(np.int64(votes[test_point]), alphas).argmax()
    
    return combined_predictions

boost_predictions = boosting_predict(X_test, all_boost_models, alphas)
boost_accuracy = np.count_nonzero(boost_predictions==np.int64(y_test))/y_test.shape[0]

print(f"Boosting Test SEt Accuracy: {boost_accuracy}")
