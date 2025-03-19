import sys, os
print(os.getcwd())
print('dir', os.path.abspath(os.getcwd(), os.pardir))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pickle
from joblib import Parallel, delayed
from pyfrechet.metric_spaces import MetricData, Sphere
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
import cloudpickle
from sklearn.preprocessing import MinMaxScaler
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
import time
from sklearn.model_selection import GridSearchCV
import base64
from scipy.stats import vonmises_fisher, vonmises_line

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])
param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'data')))
n_cores=56
n_blocks=n_samples/n_cores
current_block=int(sys.argv[1])

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
base_forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1)

M = Sphere(2)

# Function defining the regression mean m_0(theta) on S^2
def m_0(theta, mu):
    """
    Compute the regression mean on S^2.
    
    Parameters:
    theta : array-like
        Angles in [0, 2pi) that parameterize the great circle.
    mu : array-like, shape (2,)
        A unit vector defining the orientation of the great circle.
    
    Returns:
    array, shape (n, 3)
        The mean directions on S^2.
    """
    theta = np.asarray(theta)
    mu = np.asarray(mu)
    assert mu.shape == (2,) and np.isclose(np.linalg.norm(mu), 1), "mu must be a unit vector in R^2"
    
    x1 = np.cos(theta)
    x2 = np.sin(theta) * mu[0]
    x3 = np.sin(theta) * mu[1]
    
    return np.column_stack((x1, x2, x3))

# Function to generate vMF samples
def simulate_data(kappa, mu, theta_samples):
    """
    Generate samples from the von Mises-Fisher distribution on S^2.
    
    Parameters:
    sample_size : int
        Number of samples to generate.
    kappa : float
        Concentration parameter of the vMF distribution.
    mu : array-like, shape (2,)
        The unit vector defining the great circle.
    
    Returns:
    dict
        A dictionary containing input angles and generated samples.
    """
    mean_directions = m_0(theta_samples, mu)  # Compute means on S^2
    
    samples = [vonmises_fisher(mean, kappa).rvs() for mean in mean_directions]
    
    return theta_samples, np.array(samples)

# Parameters
sample_sizes = [50, 100, 200, 500]  # Sample sizes
kappa_values = [50, 200]  # Concentration parameters
mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Fixed unit vector in R^2

def tune_forest(X, y, forest = base_forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator = forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=4)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

def task(file) -> None:
    """Processes a single file for sphere data regression."""
    with open(os.path.join(os.getcwd(), 'simulations_sphere', 'data', file), 'rb') as f:
        sample = pickle.load(f)
    
    X = np.c_[sample['theta']]
    y = MetricData(M, sample['Y'].reshape(-1, 3))
    kappa = int(file.split('_')[3][5:-4])

    # Measure time for tuning and fitting
    start_time = time.time()

    # Perform hyperparameter tuning
    forest = tune_forest(X, y, base_forest, param_grid)
    oob_quantile = np.percentile(forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    end_time = time.time()
    pb_time = end_time - start_time

    ############################################################################################################
    # TYPE I COVERAGE RESULTS
    n_estimations = 50
    pb_i_cov = np.zeros(shape = (n_estimations, 3))
    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        theta = vonmises_line(kappa = 1).rvs(1) + np.pi
        # Add a column of ones for the intercept (beta_0)
        theta, new_y = simulate_data(kappa = kappa, mu = mu, theta_samples = theta)

        # Predict the new observation
        pb_new_pred = forest.predict(theta.reshape(-1,1))
        pb_i_cov[estimation, :] = (M.d(pb_new_pred, new_y) <= oob_quantile)

############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 500
    #Generate observations to estimate the probability
    theta_samples = np.array([vonmises_line(kappa = 1).rvs(MC) + np.pi]).reshape(-1, 1)
    new_theta, new_y = simulate_data(kappa = kappa, theta_samples = theta_samples, mu=mu)

    pb_new_pred = forest.predict(new_theta.reshape(-1, 1))
    pb_II_coverage = np.sum(M.d(MetricData(M, new_y), pb_new_pred) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    
############################################################################################################
    # TYPE III COVERAGE RESULTS
    pb_iii_cov = np.zeros(shape = (n_estimations, 3))

    for estimation in range(n_estimations):
        # Randomly select rows from the dataframe
        theta = np.array([np.pi/2])
        # Add a column of ones for the intercept (beta_0)
        theta, new_y = simulate_data(kappa = kappa, mu = mu, theta_samples = theta)

        # Predict the new observation
        pb_new_pred = forest.predict(theta.reshape(-1,1))
        pb_iii_cov[estimation, :] = (M.d(pb_new_pred, new_y) <= oob_quantile)

############################################################################################################
    # TYPE IV COVERAGE RESULTS
    theta = np.repeat(np.pi/2, MC)
    theta, new_y = simulate_data(kappa = kappa, theta_samples = theta, mu = mu)

    pb_new_pred = forest.predict(theta.reshape(-1,1))
    pb_IV_coverage = np.sum(M.d(MetricData(M, new_y), pb_new_pred) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC

    # Store results
    results = {
        'i_cov': pb_i_cov,
        'II_coverage': pb_II_coverage,
        'iii_cov': pb_iii_cov,
        'IV_coverage': pb_IV_coverage,
        'OOB_quantile': oob_quantile,
        'time': pb_time,
    }

    filename = os.path.join(os.getcwd(), 'simulations_sphere', 'results', f'{file[:-4]}' + str(current_block) + '_results.npy')
    np.save(filename, results)

Parallel(n_jobs=-1, verbose=40)( delayed(task)(file) for file in \
        os.listdir(os.path.join(os.getcwd(), 'simulations_sphere', 'data/'))[n_cores*(current_block-1):n_cores*(current_block)]
        if (file.endswith('.pkl'))
    )

# and not os.path.exists(os.path.join(os.getcwd(), 'simulations_sphere', 'results/' +  file[:-4]+ '_results.npy' ))
