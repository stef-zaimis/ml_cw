import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

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

# Center the x data
x_mean = np.mean(x_train)
x_centered = x_train - x_mean

model = pm.Model()

# Define the Bayesian model
with pm.Model() as model:
    # Priors for coefficients
    w0 = pm.Normal('w0', mu=0, sigma=100)
    w1 = pm.Normal('w1', mu=0, sigma=100)
    w2 = pm.Normal('w2', mu=0, sigma=100)
    w3 = pm.Normal('w3', mu=0, sigma=100)
    
    # Expected value of y given x
    y_est = w0 + w1 * x_centered + w2 * x_centered**2 + w3 * x_centered**3
    
    # Prior for noise standard deviation
    sigma = pm.HalfNormal('sigma', sigma=100)
    
    # Prior for degrees of freedom
    nu = pm.Exponential('nu', lam=1/30)
    
    # Likelihood
    y_obs = pm.StudentT('y_obs', mu=y_est, sigma=sigma, nu=nu, observed=y_train)
    
    # Inference
    trace = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.95)

    
# Summarize the posterior distributions
summary = az.summary(trace, round_to=2)
print(summary)

# Plotting the trace and posterior distributions
az.plot_trace(trace)
plt.show()

# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['y_obs'])

# Plot the model fit
# Prepare x values for prediction
x_plot = np.linspace(x_train.min(), x_train.max(), 100)
x_plot_centered = x_plot - x_mean

# Extract posterior samples
w0_samples = trace.posterior['w0'].values.flatten()
w1_samples = trace.posterior['w1'].values.flatten()
w2_samples = trace.posterior['w2'].values.flatten()
w3_samples = trace.posterior['w3'].values.flatten()

# Compute predicted y values for each posterior sample
y_pred_samples = np.zeros((len(w0_samples), len(x_plot)))

for i in range(len(w0_samples)):
    y_pred_samples[i, :] = (w0_samples[i] +
                            w1_samples[i] * x_plot_centered +
                            w2_samples[i] * x_plot_centered**2 +
                            w3_samples[i] * x_plot_centered**3)

# Compute the posterior predictive mean and credible intervals
y_pred_mean = y_pred_samples.mean(axis=0)
y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Observed data')
plt.plot(x_plot, y_pred_mean, color='red', label='Posterior predictive mean')
plt.fill_between(x_plot, y_pred_lower, y_pred_upper, color='red', alpha=0.3, label='95% credible interval')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Regression Fit with 95% Credible Interval')
plt.legend()
plt.show()

