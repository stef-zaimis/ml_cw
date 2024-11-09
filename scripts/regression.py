import numpy as np
import pandas as pd

# Load the training data
data = pd.read_csv('/home/stefanos/uni/ml/cw/regression_train.txt', sep=" ", header=None)

x_train = data[0].values
y_train = data[1].values
print(x_train, y_train)

# Code Task 10

