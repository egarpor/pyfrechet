import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
import pickle
from scipy.linalg import toeplitz

np.random.seed(1000)

def beta_calculator(j, k, dim):
    return np.sqrt(j) * np.sin(np.pi * k / (dim+1))

# Function to simulate regression data
def simulate_data(X_design, dim):
    Ys = []
    sample_size = X_design.shape[0]
    rho = 0.75
    Sigma = toeplitz(np.array([rho**i for i in range(dim)]).reshape(1,dim))
    # Generate the error term epsilon, which follows a normal distribution
    epsilon = np.random.multivariate_normal(mean=np.zeros(dim), cov=Sigma, size=sample_size) 
    # Step 2: Apply the model transformations
    X_1 = X_design[:, 0]  # X_1 corresponds to the first column (without intercept)
    X_2 = X_design[:, 1]  # X_2 corresponds to the second column
    X_3 = X_design[:, 2]  # X_3 corresponds to the third column

    # Calculate the response vector Y = beta_0 + beta_1*X_1 + beta_2*X_2 + beta_3*X_3^2 + beta_4*X_2*X_4 + beta_5*X_5 + epsilon

    n_predictors = 3

    betas = np.zeros((n_predictors, dim))
    for j in range(1, n_predictors+1):
        for k in range(1, dim+1):
            betas[j-1, k-1] = beta_calculator(j, k, dim)
    
    X = np.vstack((X_1, X_2, X_3)).T
    Ys = X @ betas + epsilon
    # Convert list to array for easier manipulation
    Ys = np.array(Ys).reshape(sample_size, dim)    
    return X_design, Ys

# Function to save the simulated regression data
def save_simulated_samples(n_samples, sample_sizes, n_predictors):
    np.random.seed(1000)
    
    # Create the folder for saving simulations if it doesn't exist
    save_folder = os.path.join(os.getcwd(), 'simulations_euc', 'volume_data')
    os.makedirs(save_folder, exist_ok=True)
    
    for dim in dims:
        for sample_size in sample_sizes:
            for k in range(1, n_samples + 1):
                # Simulate the regression data for the given sample size

                # Generate the design matrix X
                X_design = 2*np.sqrt(5)*(np.random.beta(2, 2, (sample_size, n_predictors)) - 1/2)
                
                X, Y = simulate_data(X_design=X_design, dim = dim)
                
                # Define the filename for saving
                filename = os.path.join(save_folder, f'euc_samp{k}_dim_{dim}_N{sample_size}.pkl')
                
                # Save the sample using pickle
                with open(filename, 'wb') as f:
                    pickle.dump({'X': X, 'Y': Y}, f)


dims = [1, 5, 10]  # Different dimensions for the response vector
# Set parameters for the regression scenario
n_predictors = 3  # Number of predictors

sample_sizes = [100]  # Different sample sizes

# --- Saving the simulated regression samples ---
n_samples = 500  # Number of samples to simulate and save

# Save the simulated data to files
save_simulated_samples(n_samples, sample_sizes, n_predictors)