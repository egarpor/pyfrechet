import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import pickle
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

M = H2(2)

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
    # Perform hyperparameter tuning
    best_forest = tune_forest(X, y, forest, param_grid)

    results = {
        'forest': best_forest
    }
    
    filename = os.path.join(os.getcwd(), 'simulations_H2', 'results', f'{file[:-4]}_results.npy')
    np.save(filename, results)


Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_H2', 'data/'))
    if (file.endswith('.csv') and not os.path.exists(os.path.join(os.getcwd(), 'simulations_H2', 'results/' +  file[:-4]+ '_results.npy' )))
)