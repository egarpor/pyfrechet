import sys, os
import pickle
import numpy as np
from scipy.stats import beta
import time
import joblib
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import contextlib

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])
betas = np.array([1, -1, 1])  # Define the true beta values

data_dir = os.path.join(os.getcwd(), 'sunspots', 'data')
results_dir = os.path.join(os.getcwd(), 'sunspots', 'results')
os.makedirs(results_dir, exist_ok=True)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'sunspots/' 'data')))

# Define parameter grid for tuning
param_grid = {
    'min_samples_leaf': [1, 5, 10],
    'max_features': [1, 2]
    }

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def tune_forest(X, y, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    base_forest = RandomForestRegressor(n_jobs=-1, random_state=1000, n_estimators=200, oob_score=True)
    grid_search = GridSearchCV(estimator = base_forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def task(file):    
    # Load data
    with open(os.path.join(os.getcwd(), 'sunspots', 'data', 'sunspots_data.csv'), 'rb') as f:
        sample = pd.read_csv(f)

    sample.drop(columns = ['Unnamed: 0'], inplace = True)

    train_theta_x = sample['V1'].to_numpy()[:-500]
    train_phi_x = sample['V2'].to_numpy()[:-500]
    train_theta_y = sample['V3'].to_numpy()[:-500]
    train_phi_y = sample['V4'].to_numpy()[:-500]

    test_theta_x = sample['V1'].to_numpy()[-500:]
    test_phi_x = sample['V2'].to_numpy()[-500:]
    test_theta_y = sample['V3'].to_numpy()[-500:]
    test_phi_y = sample['V4'].to_numpy()[-500:]

    train_predictors = np.vstack((train_theta_x, train_phi_x)).T
    test_predictors = np.vstack((test_theta_x, test_phi_x)).T
    # Perform hyperparameter tuning
    theta_forest = tune_forest(train_predictors, train_theta_y, param_grid) 
    phi_forest = tune_forest(train_predictors, train_phi_y, param_grid)
    # Radii
    theta_oob_quantile = np.percentile(np.abs(theta_forest.oob_prediction_ - train_theta_y), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    phi_oob_quantile = np.percentile(np.abs(phi_forest.oob_prediction_ - train_phi_y), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

############################################################################################################
    # TYPE I COVERAGE RESULTS
    n_estimations = 50
    pb_i_cov = np.zeros(shape = (n_estimations, 3))
    conf_i_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        new_X = 2*np.sqrt(5)*(np.random.beta(2, 2, (1, n_predictors)) - 1/2)
        # Add a column of ones for the intercept (beta_0)
        new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas=betas)
        new_X = new_X.reshape(1, n_predictors)

        # Predict the new observation
        pb_new_pred = pb_forest.predict(new_X)
        conf_new_pred = conf_forest.predict(new_X)

        pb_i_cov[estimation, :] = (np.abs(pb_new_pred - new_y) <= oob_quantile)
        conf_i_cov[estimation, :] = (np.abs(conf_new_pred - new_y) <= quantile)
            

############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 500
    #Generate observations to estimate the probability

    theta_new_pred = theta_forest.predict(test_predictors)
    phi_new_pred = phi_forest.predict(test_predictors)

    theta_ii_cov = np.sum(np.abs(theta_new_pred - test_theta_y).reshape(-1,1) <= np.tile(theta_oob_quantile, (MC, 1)), axis = 0) / MC
    phi_ii_cov = np.sum(np.abs(phi_new_pred - test_phi_y).reshape(-1,1) <= np.tile(phi_oob_quantile, (MC, 1)), axis = 0) / MC

############################################################################################################
    q_25 = 2*np.sqrt(5)*(beta(2,2).ppf(.25)-1/2)
    # TYPE III COVERAGE RESULTS
    pb_iii_cov = np.zeros(shape = (n_estimations, 3))
    conf_iii_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        new_X = np.repeat(q_25, n_predictors).reshape(1, n_predictors)
        # Add a column of ones for the intercept (beta_0)
        new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas=betas)

        # Predict the new observation
        pb_new_pred = pb_forest.predict(new_X)
        conf_new_pred = conf_forest.predict(new_X)

        pb_iii_cov[estimation, :] = (np.abs(pb_new_pred - new_y) <= oob_quantile)
        conf_iii_cov[estimation, :] = (np.abs(conf_new_pred - new_y) <= quantile)

############################################################################################################
    # TYPE IV COVERAGE RESULTS
    #Generate observations to estimate the probability
    new_X = np.repeat(q_25, MC * n_predictors).reshape(MC, n_predictors)
    new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas = betas)

    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)

    pb_iv_cov = np.sum(np.abs(pb_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    conf_iv_cov = np.sum(np.abs(conf_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(quantile, (MC, 1)), axis = 0) / MC

    # Store results
    results = {
        'pb_i_cov': pb_i_cov,
        'conf_i_cov': conf_i_cov,
        'pb_ii_cov': pb_ii_cov,
        'conf_ii_cov': conf_ii_cov,
        'pb_iii_cov': pb_iii_cov,
        'conf_iii_cov': conf_iii_cov,
        'pb_iv_cov': pb_iv_cov,
        'conf_iv_cov': conf_iv_cov,
        'OOB_quantile': oob_quantile,
        'quantile': quantile,
        'pb_time': pb_time,
        'conf_time': conf_time,
    }
    filename = os.path.join(results_dir, file[:-4] + '_results.npy')
    np.save(filename, results)

file_list = list(filter(lambda file: file.endswith(f'block_{current_block}.pkl'), filter(lambda file: file.endswith('.pkl'), os.listdir(os.path.join(os.getcwd(), 'sunspots', 'data/')))))
total_files = len(file_list)

with tqdm_joblib(tqdm(desc="Percentage of tasks completed:", total = total_files)) as progress_bar:
    Parallel(n_jobs=-1, verbose=2)( delayed(task)(file) for file in file_list)