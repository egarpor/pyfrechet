import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import pickle
import joblib
from joblib import Parallel, delayed
from pyfrechet.metric_spaces import MetricData, H2
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
import cloudpickle
from sklearn.preprocessing import MinMaxScaler
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import base64
from scipy.stats import vonmises_fisher
from tqdm import tqdm
import contextlib

np.random.seed(1000)
param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_H2/' 'data')))
current_block=int(sys.argv[1])

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
base_forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1)

M = H2(2)

def tune_forest(X, y, forest = base_forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=0)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

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

# Define metric space

def task(file) -> None:
    """Processes a single file for hyperboloid data regression."""
    with open(os.path.join(os.getcwd(), 'simulations_H2', 'data', file), 'rb') as f:
        sample = pd.read_csv(f)
        
    sample.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 4 as V1, V2 and V3
    sample.columns = ['t', 'V1', 'V2', 'V3']

    X = sample['t'].values.reshape(-1, 1)
    y = MetricData(M, sample.iloc[:, 1:4].values)
    kappa = int(file.split('_')[3][5:])
    N = int(file.split('_')[2][1:])
    samp = int(file.split('_')[1][4:])
    # Perform hyperparameter tuning
    forest = tune_forest(X, y, base_forest, param_grid)
    oob_quantile = np.percentile(forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    
    ############################################################################################################
    # TYPE I COVERAGE RESULTS
    n_estimations = 50
    pb_i_cov = np.zeros(shape = (n_estimations, 3))
    for estimation in range(n_estimations):
        filename = os.path.join(os.getcwd(), 'simulations_H2/TypeIdata', f'estimation_{estimation+1}_N{N}_kappa{kappa}.csv')
        hyp_data = pd.read_csv(filename)
        hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
        hyp_data.columns = ['t', 'V1', 'V2', 'V3']
        hyp_data = hyp_data.iloc[int(file.split('_')[1][4:]) - 1]
        theta = hyp_data['t']
        new_y = hyp_data[['V1', 'V2', 'V3']]

        # Predict the new observation
        pb_new_pred = forest.predict(theta.reshape(-1,1))
        pb_i_cov[estimation, :] = (M.d(pb_new_pred, new_y) <= oob_quantile)

############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 500
    filename = os.path.join(os.getcwd(), 'simulations_H2/TypeIIdata', f'samp_{samp}_N_{N}_kappa{kappa}.csv')
    hyp_data = pd.read_csv(filename)
    hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 3 as V1, V2 and V3
    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    thetas = hyp_data['t'].values
    new_ys = hyp_data[['V1', 'V2', 'V3']].values
    pb_new_preds = forest.predict(thetas.reshape(-1, 1))
    pb_ii_cov = np.sum(M.d(new_ys, pb_new_preds.data).reshape(MC,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC
    
############################################################################################################
    # TYPE III COVERAGE RESULTS
    pb_iii_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        filename = os.path.join(os.getcwd(), 'simulations_H2/TypeIIIdata', f'estimation_{estimation+1}_N{N}_kappa{kappa}.csv')
        hyp_data = pd.read_csv(filename)
        hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
        hyp_data.columns = ['t', 'V1', 'V2', 'V3']
        hyp_data = hyp_data.iloc[int(file.split('_')[1][4:]) - 1]
        theta = hyp_data['t']
        new_y = hyp_data[['V1', 'V2', 'V3']]

        # Predict the new observation
        pb_new_pred = forest.predict(theta.reshape(-1,1))
        pb_iii_cov[estimation, :] = (M.d(pb_new_pred, new_y) <= oob_quantile)

############################################################################################################
    # TYPE IV COVERAGE RESULTS
    filename = os.path.join(os.getcwd(), 'simulations_H2/TypeIVdata', f'samp_{samp}_N_{N}_kappa{kappa}.csv')
    hyp_data = pd.read_csv(filename)
    hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 3 as V1, V2 and V3
    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    thetas = hyp_data['t'].values
    new_ys = hyp_data[['V1', 'V2', 'V3']].values

    pb_new_preds = forest.predict(thetas.reshape(-1,1))
    pb_iv_cov = np.sum(M.d(new_ys, pb_new_preds.data).reshape(MC,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC

    # Store results
    results = {
        'i_cov': pb_i_cov,
        'ii_cov': pb_ii_cov,
        'iii_cov': pb_iii_cov,
        'iv_cov': pb_iv_cov,
        'OOB_quantile': oob_quantile
        }
    results_filename = os.path.join(os.getcwd(), 'simulations_H2', 'results', f'{file[:-4]}' + '_block_' + str(current_block) + '_results.npy')
    np.save(results_filename, results)

file_list = list(filter(lambda file: file.endswith(f'block_{current_block}.csv'), filter(lambda file: file.endswith('.csv'), os.listdir(os.path.join(os.getcwd(), 'simulations_H2', 'data/')))))

with tqdm_joblib(tqdm(desc="Percentage of tasks completed:", total=56)) as progress_bar:

    Parallel(n_jobs=-1, verbose=2)( delayed(task)(file) for file in \
            # select files that end with "block_{current_block}.pkl"
            file_list[0:56]
        )
