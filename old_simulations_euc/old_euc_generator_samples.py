import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
import pickle

np.random.seed(1000)

# Function to simulate multivariate regression data
def simulate_data(q, sigma, n_predictors, sample_size, beta):
    X_designs = []
    Ys = []
    
    for _ in range(sample_size):
        # Generate the design matrix X, adding the column of ones for the intercept term
        X_design = np.random.uniform(0, 1, size=(1, n_predictors))
        
        # Generate the error term epsilon, which follows a multivariate normal distribution
        epsilon = np.random.multivariate_normal(np.zeros(q), sigma * np.eye(q))
        
        # Calculate the response vector Y = Beta * X + epsilon
        Y = X_design @ beta + epsilon
        
        X_designs.append(X_design)
        Ys.append(Y)
        
    # Convert lists to arrays for easier manipulation
    X_designs = np.array(X_designs).reshape(sample_size, n_predictors)
    Ys = np.array(Ys).reshape(sample_size, q)
    
    # Center the predictors
    X_designs -= X_designs.mean(axis=0)
    
    # Center the responses
    Ys -= Ys.mean(axis=0)
    
    return X_designs, Ys


# Function to save the simulated regression data
def save_simulated_samples(q, n_samples, sample_sizes, n_predictors, sigma_values, beta):
    np.random.seed(1000)
    
    # Create the folder for saving simulations if it doesn't exist
    save_folder = os.path.join(os.getcwd(), 'simulations_euc', 'data')
    os.makedirs(save_folder, exist_ok=True)
    
    for sample_size in sample_sizes:
        for sigma in sigma_values:
            for k in range(1, n_samples + 1):
                # Simulate the regression data for the given sample size and sigma
                X_design, Y = simulate_data(q=q, sigma=sigma, n_predictors = n_predictors, sample_size=sample_size, beta=beta)
                
                # Define the filename for saving
                filename = os.path.join(save_folder, f'euc_samp{k}_N{sample_size}_sigma{sigma}.pkl')
                
                # Save the sample using pickle
                with open(filename, 'wb') as f:
                    pickle.dump({'X': X_design, 'Y': Y}, f)

# Set parameters for the regression scenario
q = 3 # Dimensionality of the data (q features)
n_predictors = 5  # Number of predictors
sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different sigma values
sample_sizes = [50, 100, 200, 500]  # Different sample sizes

# Sample beta from a multivariate normal distribution
beta = np.random.multivariate_normal(np.ones(n_predictors * q), 2*np.eye(n_predictors * q)).reshape(n_predictors, q) # n_predictors*q

# --- Saving the simulated regression samples ---
n_samples = 100  # Number of samples to simulate and save

# Save the simulated data to files
save_simulated_samples(q, n_samples, sample_sizes, n_predictors, sigma_values, beta)
