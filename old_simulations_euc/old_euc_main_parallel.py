import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

import pickle 
import numpy as np
from joblib import Parallel, delayed
from pyfrechet.metric_spaces import MetricData, Euclidean
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree

from pyfrechet.metrics import mse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed


# By-blocks execution
n_samples = len(os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data')))
n_cores = 56
n_blocks = n_samples / n_cores
current_block = int(sys.argv[1])

# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 3, 5, 7, 10, 15]
}

# Custom scorer (negative mean squared error, assuming mse is defined elsewhere)
neg_mse = make_scorer(mse, greater_is_better=False)

def tune_forest(X, y):
    """ Perform hyperparameter tuning using GridSearchCV. """
    base = Tree(split_type='2means', impurity_method='cart', mtry = 5)
    forest = BaggedRegressor(estimator=base, n_estimators=100, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=1)
    
    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=-1, verbose=4)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

def task(file) -> None:
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_euc', 'data/' + file), 'rb') as f:
        sample = pickle.load(f)
    # Convert response to MetricData
    M = Euclidean(dim=15)
    X = sample['X']
    unique_rows, counts = np.unique(X, axis=0, return_counts=True)
    if np.any(counts > 1):
        print("There are repetitions in the rows.")
    else:
        pass
    y = MetricData(M, sample['Y'])

    # Perform hyperparameter tuning
    best_forest = tune_forest(X, y)
    
    # Fit the best forest
    best_forest.fit(X, y)
    
    results = {
        'x_train_data': X,
        'y_train_data': y,
        'train_predictions': best_forest.predict(X),
        'forest': best_forest,
    }
    
    filename = 'simulations_euc/results/' + '_' + file[:-4] + '_block_' + str(current_block) + '_results'
    np.save(filename, results)

print(f'Block number: {current_block}')

# One sample per core in the current block
Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data/'))[n_cores * (current_block - 1):n_cores * current_block]
    if file.endswith('.pkl')
)