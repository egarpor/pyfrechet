import os
import pickle
import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

results_dir = os.path.join(os.getcwd(), 'simulations_euc', 'weekly')
os.makedirs(results_dir, exist_ok=True)

# Define parameter grid for tuning
param_grid = {
    'min_samples_leaf': [1, 5, 10],
    'max_features': [1, 2, 3]
    }

def tune_forest(X, y, param_grid, n_trees):
    """Perform hyperparameter tuning using GridSearchCV."""
    base_forest = RandomForestRegressor(n_jobs=1, random_state=1000, n_estimators=n_trees, max_features=1, oob_score=True)
    grid_search = GridSearchCV(estimator=base_forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=4)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
    
# Function to simulate regression data
def simulate_data(sigma, X_design, betas):
    Ys = []
    sample_size = X_design.shape[0]

    # Generate the error term epsilon, which follows a normal distribution
    epsilon = np.random.normal(0, sigma, size = sample_size).reshape(sample_size, 1)

    # Step 2: Apply the model transformations
    X_1 = X_design[:, 0]  # X_1 corresponds to the first column (without intercept)
    X_2 = X_design[:, 1]  # X_2 corresponds to the second column
    X_3 = X_design[:, 2]  # X_3 corresponds to the third column

    # Calculate the response vector Y = beta_0 + beta_1*X_1 + beta_2*X_2 + beta_3*X_3 + epsilon
    Ys = betas[0] * X_1 + betas[1] * X_2 + betas[2] * X_3
    # Convert list to array for easier manipulation
    Ys = Ys.reshape(sample_size, 1) + epsilon
    
    return X_design, Ys

# Function to save the simulated regression data
np.random.seed(1000)
MC = 2000

betas = np.array([1, -1, 1])  # Define the true beta values

# Set parameters for the regression scenario
n_predictors = 3  # Number of predictors
sigma_values = [np.sqrt(3)/2]  # Different sigma values

def task(sample_size, n_trees, sigma, k):
# Create the folder for saving simulations if it doesn't exist
    # Generate the design matrix X
    X_design = np.sqrt(2)*2*(np.random.beta(1/2, 1/2, (sample_size, n_predictors)) - 1/2)
    X, Y = simulate_data(sigma=sigma, X_design=X_design, betas=betas)
    y = Y.ravel()

    # PREDICTION BALLS
    pb_forest = tune_forest(X, y, param_grid, n_trees)
    oob_quantile = np.percentile(np.abs(pb_forest.oob_prediction_ - y), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    # SPLIT-CONFORMAL
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    conf_forest = tune_forest(X_train, y_train, param_grid, n_trees)
    test_preds = conf_forest.predict(X_test)
    quantile = np.percentile(np.abs(y_test - test_preds), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    # TYPE I
    new_X_design = np.sqrt(2)*2*(np.random.beta(1/2, 1/2, (1, n_predictors)) - 1/2)
    _, new_y = simulate_data(sigma = sigma, X_design=new_X_design, betas=betas)
    new_X = new_X_design.reshape(1, n_predictors)

    #Predict the new observation
    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)
    # Store the selected values
    pb_yesno = np.abs(pb_new_pred - new_y) <= oob_quantile
    conf_yesno = np.abs(conf_new_pred - new_y) <= quantile

    # Store results
    results = {'n_trees': n_trees,
    'pb_radius': oob_quantile,
    'conf_radius': quantile,
    'pb_yesno': pb_yesno,
    'conf_yesno': conf_yesno
    }
    
    filename = os.path.join(results_dir, f'weekly_samp{k}_trees_{n_trees}_N_{sample_size}_sigma_{np.round(sigma, 1)}')
    np.save(filename, results)
    
for n_trees in [100, 200, 500, 1000]:
    for sample_size in [200]:
        for sigma in sigma_values:
            Parallel(n_jobs = 12, verbose = 40)(
                delayed(task)(sample_size, n_trees, sigma, k)
                for k in range(1, MC+1)
                if (not os.path.exists(os.path.join(results_dir, f'weekly_samp{k}_trees_{n_trees}_N_{sample_size}_sigma_{np.round(sigma, 1)}.pkl.npy')))
            )