import sys, os
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
from scipy.stats import vonmises_fisher

def lambda2str(expr):
    b = cloudpickle.dumps(expr)
    s = base64.b64encode(b).decode()
    return s

def str2lambda(s):
    b = base64.b64decode(s)
    expr = cloudpickle.loads(b)
    return expr

param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
forest = BaggedRegressor(estimator=base, n_estimators=100, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1)

def tune_forest(X, y, forest=forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=4)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

M = Sphere(2)

# Define metric space

def task(file) -> None:
    """Processes a single file for sphere data regression."""
    with open(os.path.join(os.getcwd(), 'simulations_sphere', 'data', file), 'rb') as f:
        sample = pickle.load(f)
    
    X = np.c_[sample['theta']]
    y = MetricData(M, sample['Y'].reshape(-1, 3))
    # Measure time for tuning and fitting
    start_time = time.time()

    # Perform hyperparameter tuning
    best_forest = tune_forest(X, y, forest, param_grid)

    end_time = time.time()
    tuning_fitting_time = end_time - start_time
    # s = lambda2str(best_forest)      
    # e2 = str2lambda(s)
    results = {
        'forest': best_forest
    }
    
    filename = os.path.join(os.getcwd(), 'simulations_sphere', 'results', f'{file[:-4]}_results.npy')
    np.save(filename, results)

Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_sphere', 'data/'))
    if (file.endswith('.pkl') and not os.path.exists(os.path.join(os.getcwd(), 'simulations_sphere', 'results/' +  file[:-4]+ '_results.npy' )))
)