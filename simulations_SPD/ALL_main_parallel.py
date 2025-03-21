import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle 
import numpy as np

from pyfrechet.metrics import mse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed

from pyfrechet.metric_spaces import MetricData, Euclidean, CustomAffineInvariant, LogCholesky, spd_to_log_chol
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree

# Parameters
sample_sizes = [50, 100, 200, 500]  # Sample sizes
dfs = [5, 15]
sign_level = np.array([0.01, 0.05, 0.1])

# Define the matrices to interpolate 
Sigma_1 = np.array([[1, -0.6],
                  [-0.6, 0.5]])
Sigma_2 = np.array([[1, 0],
                  [0, 1]])
Sigma_3 = np.array([[0.5, 0.4],
                  [0.4, 1]])

# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error
neg_mse = make_scorer(mse, greater_is_better=False)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data')))
current_block = int(sys.argv[1])

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
base_forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1)

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

def Sigma_t(t_array, Sigma_array):
    """Provides an array with the matrices given by a regression model that interpolates between four matrices."""  
    """The regression starts with Sigma_1 and then goes to Sigma_2 and Sigma_3 and ends in Sigma_4."""
    
    # Define time intervals for interpolation
    t_array = np.array(t_array)
    t_array = t_array[:, None, None]

    # Return the interpolated matrices
    return np.where(t_array < 0.5, np.cos(np.pi*t_array)**2 * Sigma_array[0] + (1 - np.cos(np.pi*(1-t_array))**2) * Sigma_array[1], 0) + np.where(t_array >= 0.5, (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1] + np.cos(np.pi*(1-t_array))**2 * Sigma_array[2], 0)

def sim_regression_matrices(Sigmas: tuple,
                            t: np.array,
                            df: int=2):
    t = np.array(t)
    
    q = Sigmas[0].shape[0]

    c_dq = 2 * np.exp((1 / q) * sum( digamma((df - np.arange(1, q + 1) + 1 ) / 2) ))
    sigma_t = Sigma_t(t, Sigmas)
    sample_Y = [wishart( df=df, scale = sigma_t[k]/c_dq ).rvs( size=1 ) for k in range(t.shape[0])]
    return {'t': t, 'y': sample_Y} 

def tune_forest(X, y, forest = base_forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator = forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=0)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

def task(file) -> None:
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_SPD', 'data/' + file), 'rb') as f:
        sample = pickle.load(f)
    # Read the data

    X=np.c_[sample['t']]
    sample_Y = np.array(sample['y'])
    df = int(file.split('_')[3][2:])
    for dist in ['LC', 'AI', 'LE']:
        if dist == 'LC':
            M_lc = LogCholesky(dim=2)
            sampleY_LogChol = np.c_[[spd_to_log_chol(A) for A in sample['y']]]
            y_lc = MetricData(M_lc, sampleY_LogChol)
            
        elif dist == 'AI':
            M_ai = CustomAffineInvariant(dim=2)
            y_ai = MetricData(M_ai, sample_Y)
            
        elif dist == 'LE':
            M_le = LogEuclidean(dim=2)
            y_le = MetricData(M_le, sample_Y)

    # Perform hyperparameter tuning
    forest_ai = tune_forest(X, y_ai, base_forest, param_grid)
    oob_quantile_ai = np.percentile(forest_ai.oob_errors_matrix(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    forest_lc = tune_forest(X, y_lc, base_forest, param_grid)
    oob_quantile_lc = np.percentile(forest_lc.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    forest_le = tune_forest(X, y_le, base_forest, param_grid)
    oob_quantile_le = np.percentile(forest_le.oob_errors_matrix(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    ############################################################################################################
    # TYPE I COVERAGE RESULTS
    n_estimations = 50
    ai_i_cov = np.zeros(shape = (n_estimations, 3))
    lc_i_cov = np.zeros(shape = (n_estimations, 3))
    le_i_cov = np.zeros(shape = (n_estimations, 3))
    for estimation in range(n_estimations):
        # Generate a new observation
        new_t = np.random.uniform(size = 1)
        new_y = sim_regression_matrices(Sigmas = (Sigma_1, Sigma_2, Sigma_3), 
                                         t = new_ts,  
                                         df = df)['y']
        # Create MetricData objects
        new_y_logchol = np.c_[[spd_to_log_chol(A) for A in new_y]]
        new_y_logchol = MetricData(M_lc, new_y_logchol)
        new_y_ai = MetricData(M_ai, new_y)
        new_y_le = MetricData(M_le, new_y)
        # Predict the new observation
        ai_new_pred = forest_ai.predict(new_t.reshape(-1,1))
        lc_new_pred = forest_lc.predict(new_t.reshape(-1,1))
        le_new_pred = forest_le.predict(new_t.reshape(-1,1))

        ai_i_cov[estimation, :] = (M_ai.d(ai_new_pred, new_y) <= oob_quantile)
        lc_i_cov[estimation, :] = (M_lc.d(lc_new_pred, new_y) <= oob_quantile)
        le_i_cov[estimation, :] = (M_le.d(le_new_pred, new_y) <= oob_quantile)

############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 500
    #Generate observations to estimate the probability
    new_ts = np.random.uniform(size = MC)
    new_ys = sim_regression_matrices(Sigmas = (Sigma_1, Sigma_2, Sigma_3), 
                                    t = new_t,  
                                    df = ln['df'])['y']
    # Create MetricData objects
    new_y_logchol = np.c_[[spd_to_log_chol(A) for A in new_y]]
    new_y_logchol = MetricData(M_lc, new_y_logchol)
    new_y_ai = MetricData(M_ai, new_y)
    new_y_le = MetricData(M_le, new_y)
    # Predict the new observation
    ai_new_pred = forest_ai.predict(new_t.reshape(-1,1))
    lc_new_pred = forest_lc.predict(new_t.reshape(-1,1))
    le_new_pred = forest_le.predict(new_t.reshape(-1,1))
    
    ai_ii_cov = np.sum(M_ai.d(ai_new_pred, new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC
    lc_ii_cov = np.sum(M_lc.d(lc_new_pred, new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC
    le_i_cov = np.sum(M_le.d(le_new_pred, new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC

############################################################################################################
    # TYPE III COVERAGE RESULTS

    ai_i_cov = np.zeros(shape = (n_estimations, 3))
    lc_i_cov = np.zeros(shape = (n_estimations, 3))
    le_i_cov = np.zeros(shape = (n_estimations, 3))
    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        theta = np.array([vonmises_line.ppf(q=0.25, kappa = 1)])
        # Add a column of ones for the intercept (beta_0)
        theta, new_y = simulate_data(kappa = kappa, mu = mu, theta_samples = theta)

        # Predict the new observation
        pb_new_pred = forest.predict(theta.reshape(-1,1))
        pb_iii_cov[estimation, :] = (M.d(pb_new_pred, new_y) <= oob_quantile)


Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data/'))[n_cores*(current_block-1):n_cores*(current_block)]
    if (file.endswith('.pkl')) #and not os.path.exists(os.path.join(os.getcwd(), 'simulations_SPD', 'results/' + 'WASS_Samp' +  file[9:-4]+ '_results.npy' )))
)