import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch import from_numpy
from torch import no_grad

# Load the training data
data = pd.read_csv('/home/stefanos/uni/ml/cw/regression_train.txt', sep=" ", header=None)

x_train = data[0].values
x_reshaped = x_train.reshape(-1,1)
y_train = data[1].values
#print(x_train, y_train)

# Code Task 10
lin_reg = LinearRegression()
lin_reg.fit(x_reshaped, y_train)
print("Coefficient (slope):", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)

# Code Task 11
x_train_tensor = from_numpy(x_reshaped).float()
y_train_tensor = from_numpy(y_train).float().view(-1, 1)

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

nn_model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimiser = Adam(nn_model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    nn_model.train()
    for batch_x, batch_y in train_dataloader:
        predictions = nn_model(batch_x)
        loss = loss_fn(predictions, batch_y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if (epoch+1)%50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Code Task 12
x_centered = x_train - np.mean(x_train)
num_samples=1000
reg_model = pm.Model()

with reg_model:
    w0 = pm.Normal('w0', mu=0, sigma=100)
    w1 = pm.Normal('w1', mu=0, sigma=100)
    w2 = pm.Normal('w2', mu=0, sigma=100)
    w3 = pm.Normal('w3', mu=0, sigma=100)

    sigma = pm.HalfNormal('sigma', sigma=100)
    
    y_est = w0 + w1 * x_centered + w2 * x_centered**2 + w3 * x_centered**3
    
    nu = pm.Exponential('nu', lam=1/30)
    
    likelihood = pm.StudentT('y_obs', mu=y_est, sigma=sigma, nu=nu, observed=y_train)
    
    #sampler = pm.NUTS()
    idata = pm.sample(num_samples, tune=2000, return_inferencedata=True, target_accept=0.95, progressbar=True) # This is slow

print(az.summary(idata, round_to=2))

az.plot_trace(idata)
plt.show()


# Code Task 13:
test_data = pd.read_csv('/home/stefanos/uni/ml/cw/regression_test.txt', sep=" ", header=None)
x_test = test_data[0].values.reshape(-1,1)
y_test = test_data[1].values

linreg_pred = lin_reg.predict(x_test)
mse_linreg = mean_squared_error(y_test, linreg_pred)
print("Linear Regression Test MSE:", mse_linreg)

x_test_tensor = from_numpy(x_test).float()

nn_model.eval()
with no_grad():
    y_pred_nn = nn_model(x_test_tensor).numpy()

mse_nn = mean_squared_error(y_test, y_pred_nn)
print("Neural Network Test MSE:", mse_nn)

