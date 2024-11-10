import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

# Code Task 11
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

nn = NeuralNetwork()
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_dataloader:
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if (epoch+1)%50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
