import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the training data
data = pd.read_csv('/home/stefanos/uni/ml/cw/regression_train.txt', sep=" ", header=None)

x_train = data[0].values.reshape(-1, 1)
y_train = data[1].values
#print(x_train, y_train)

# Code Task 10
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print("Coefficient (slope):", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
