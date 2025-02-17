import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
import pickle

np.random.seed(1000)

# Function to simulate multivariate regression data
def simulate_data(sigma, X_design, betas):
    Ys = []
    sample_size = X_design.shape[0]

    # Generate the error term epsilon, which follows a multivariate normal distribution
    epsilon = np.random.normal(0, sigma, size = sample_size)
    
    # Step 2: Apply the model transformations
    X_1 = X_design[:, 1]  # X_1 corresponds to the first column (without intercept)
    X_2 = X_design[:, 2]  # X_2 corresponds to the second column
    X_3 = X_design[:, 3]  # X_3 corresponds to the third column
    X_4 = X_design[:, 4]  # X_4 corresponds to the fourth column
    X_5 = X_design[:, 5]  # X_5 corresponds to the fifth column

    # Calculate the response vector Y = beta_0 + beta_1*X_1 + beta_2*sin(pi*X_2) + beta_3*X_3^2 + beta_4*X_2*X_4 + beta_5*X_5 + epsilon
    sin_X2 = np.sin(np.pi * X_2)  # Apply sin to X_2
    X_3_squared = X_3 ** 2  # Apply square to X_3
    X_2_X_4 = X_2 * X_4  # Interaction term between X_2 and X_4

    Ys = betas[0] + betas[1] * X_1 + betas[2] * sin_X2 + betas[3] * X_3_squared + betas[4] * X_2_X_4 + betas[5] * X_5 + epsilon
            
    # Convert list to array for easier manipulation
    Ys = np.array(Ys).reshape(sample_size, 1)
    
    # Center the predictors
    #X_design -= X_design.mean(axis=0)
    
    # Center the responses
    #Ys -= Ys.mean(axis=0)
    
    return X_design[:,1:], Ys

# Function to save the simulated regression data
def save_simulated_samples(n_samples, sample_sizes, n_predictors, sigma_values, betas):
    np.random.seed(1000)
    
    # Create the folder for saving simulations if it doesn't exist
    save_folder = os.path.join(os.getcwd(), 'simulations_euc', 'data')
    os.makedirs(save_folder, exist_ok=True)
    
    for sample_size in sample_sizes:
        for sigma in sigma_values:
            for k in range(1, n_samples + 1):
                # Simulate the regression data for the given sample size and sigma

                # Generate the design matrix X
                X_design = np.random.uniform(0, 1, size=(sample_size, n_predictors))
                # Add a column of ones for the intercept (beta_0)
                X_design = np.c_[np.ones(sample_size), X_design]

                X, Y = simulate_data(sigma=sigma, X_design=X_design, betas=betas)
                
                # Define the filename for saving
                filename = os.path.join(save_folder, f'euc_samp{k}_N{sample_size}_sigma{sigma}.pkl')
                
                # Save the sample using pickle
                with open(filename, 'wb') as f:
                    pickle.dump({'X': X, 'Y': Y}, f)


betas = np.array([-1, -1, 2, -3, 3, -1])  # Define the true beta values

# Set parameters for the regression scenario
n_predictors = 5  # Number of predictors
sigma_values = [0.1, 0.5, 1]  # Different sigma values
sample_sizes = [50, 100, 200, 500]  # Different sample sizes

# --- Saving the simulated regression samples ---
n_samples = 500  # Number of samples to simulate and save

# Save the simulated data to files
save_simulated_samples(n_samples, sample_sizes, n_predictors, sigma_values, betas = betas)