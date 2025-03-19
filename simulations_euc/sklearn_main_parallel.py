import os
import pickle
import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])
betas = np.array([1, -1, 1])  # Define the true beta values

data_dir = os.path.join(os.getcwd(), 'simulations_euc', 'data')
results_dir = os.path.join(os.getcwd(), 'simulations_euc', 'results')
os.makedirs(results_dir, exist_ok=True)

# Define parameter grid for tuning
param_grid = {
    'min_samples_leaf': [1, 5, 10],
    'max_features': [1, 2, 3]
    }

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

def tune_forest(X, y, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    base_forest = RandomForestRegressor(n_jobs=1, random_state=1000, n_estimators=2000, max_features=1, oob_score=True)
    grid_search = GridSearchCV(estimator=base_forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=4)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def task(file):    
    # Load data
    with open(os.path.join(data_dir, file), 'rb') as f:
        sample = pickle.load(f)

    sigma_approx = float(file.split('_')[3][5:-4])
    if sigma_approx == 0.6:
        true_sigma = 1/np.sqrt(3)
    elif sigma_approx == 0.9:
        true_sigma = np.sqrt(3)/2
    elif sigma_approx == 1.7:
        true_sigma = np.sqrt(3)
    else:
        raise ValueError("Sigma value not found.")

    X = sample['X']
    y = sample['Y']
    y = y.ravel()
    n_predictors = X.shape[1]

    # Conformal data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    # Measure time for tuning and fitting (prediction balls)
    start_time = time.time()
    # Perform hyperparameter tuning
    pb_forest = tune_forest(X, y, param_grid) 
    # Radii   
    oob_quantile = np.percentile(np.abs(pb_forest.oob_prediction_ - y), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    end_time = time.time()
    pb_time = end_time - start_time
    print(f"Finished pb tuning {file}.")
    # Measure time for tuning and fitting (SC regions)
    start_time = time.time()
    # Perform hyperparameter tuning
    conf_forest = tune_forest(X_train, y_train, param_grid) 
    print(f"Finished conf tuning {file}.")
    # Radii 
    test_preds = conf_forest.predict(X_test)
    quantile = np.percentile(np.abs(y_test - test_preds), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    end_time = time.time()
    conf_time = end_time - start_time
    print(f"Finished tuning {file}.")
############################################################################################################
    # TYPE I COVERAGE RESULTS
    n_estimations = 50
    pb_i_cov = np.zeros(shape = (n_estimations, 3))
    conf_i_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        new_X = np.sqrt(2)*2*(np.random.beta(1/2, 1/2, (1, n_predictors)) - 1/2)
        # Add a column of ones for the intercept (beta_0)
        new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas=betas)
        new_X = new_X.reshape(1, n_predictors)

        # Predict the new observation
        pb_new_pred = pb_forest.predict(new_X)
        conf_new_pred = conf_forest.predict(new_X)

        pb_i_cov[estimation, :] = (np.abs(pb_new_pred - new_y) <= oob_quantile)
        conf_i_cov[estimation, :] = (np.abs(conf_new_pred - new_y) <= quantile)
            
    print(f"Finished type I coverage {file}.")
############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 500
    #Generate observations to estimate the probability
    new_X = np.sqrt(2)*2*(np.random.beta(1/2, 1/2, (MC, n_predictors)) - 1/2)
    new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas = betas)

    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)

    pb_II_coverage = np.sum(np.abs(pb_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    conf_II_coverage = np.sum(np.abs(conf_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(quantile, (MC, 1)), axis = 0) / MC
    print(f"Finished type II coverage {file}.")
############################################################################################################
    # TYPE III COVERAGE RESULTS
    pb_iii_cov = np.zeros(shape = (n_estimations, 3))
    conf_iii_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        new_X = np.repeat(-1, n_predictors).reshape(1, n_predictors)
        # Add a column of ones for the intercept (beta_0)
        new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas=betas)

        # Predict the new observation
        pb_new_pred = pb_forest.predict(new_X)
        conf_new_pred = conf_forest.predict(new_X)

        pb_iii_cov[estimation, :] = (np.abs(pb_new_pred - new_y) <= oob_quantile)
        conf_iii_cov[estimation, :] = (np.abs(conf_new_pred - new_y) <= quantile)
    print(f"Finished type III coverage {file}.")
############################################################################################################
    # TYPE IV COVERAGE RESULTS
    #Generate observations to estimate the probability
    new_X = np.repeat(-1, MC * n_predictors).reshape(MC, n_predictors)
    new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas = betas)

    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)

    pb_IV_coverage = np.sum(np.abs(pb_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    conf_IV_coverage = np.sum(np.abs(conf_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(quantile, (MC, 1)), axis = 0) / MC
    print(f"Finished type IV coverage {file}.")

    # Store results
    results = {
        'pb_i_cov': pb_i_cov,
        'conf_i_cov': conf_i_cov,
        'pb_II_coverage': pb_II_coverage,
        'conf_II_coverage': conf_II_coverage,
        'pb_iii_cov': pb_iii_cov,
        'conf_iii_cov': conf_iii_cov,
        'pb_IV_coverage': pb_IV_coverage,
        'conf_IV_coverage': conf_IV_coverage,
        'OOB_quantile': oob_quantile,
        'quantile': quantile,
        'pb_time': pb_time,
        'conf_time': conf_time,
    }
    print(f"Finished {file}.")
    filename = os.path.join(results_dir, file[:-4] + '_results.npy')
    print(filename)
    np.save(filename, results)

Parallel(n_jobs=2, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(data_dir)
    if (file.endswith('.pkl') and not os.path.exists(os.path.join(os.getcwd(), 'simulations_euc', 'results/' +  file[:-4]+ '_results.npy' )))
)