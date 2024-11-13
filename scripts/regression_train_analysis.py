import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
data = pd.read_csv('regression_train.txt', sep=" ", header=None, names=['x', 'y'])

x_train = data['x'].values
y_train = data['y'].values

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of Training Data')
plt.legend()
plt.show()

