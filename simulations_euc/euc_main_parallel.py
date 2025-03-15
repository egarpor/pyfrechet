import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from pyfrechet.metric_spaces import MetricData, Euclidean
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import time

# By-blocks execution
# n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data')))
#n_cores=8
##n_cores=int(input('Introduce number of cores: '))
#n_blocks = n_samples/n_cores
#current_block = int(sys.argv[1])

M = two_euclidean(dim=1)
# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 5, 10, 20]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

base = Tree(split_type='2means', mtry=2, impurity_method='cart')
forest = BaggedRegressor(estimator=base, n_estimators=100, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=1)
    
def tune_forest(X, y, forest=forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """

    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=4)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

def task(file) -> None:    
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_euc', 'data', file), 'rb') as f:
        sample = pickle.load(f)
    
    # Define metric space
    X = sample['X']
    y = MetricData(M, sample['Y'])

    # Measure time for tuning and fitting
    start_time = time.time()

    # Perform hyperparameter tuning
    best_forest = tune_forest(X, y, forest, param_grid)

    end_time = time.time()
    tuning_fitting_time = end_time - start_time
    # Store results
    results = {
        'forest': best_forest,
        'tuning_fitting_time': tuning_fitting_time
    }
    
    filename = os.path.join(os.getcwd(), 'simulations_euc', 'results', file[:-4] + '_results.npy')
    np.save(filename, results)


Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data/'))
    if (file.endswith('.pkl')) and not os.path.exists(os.path.join(os.getcwd(), 'simulations_euc', 'results/' + 'euc_samp' +  file[8:-4]+ '_results.npy' ) and (int(file.split('_')[1][4:]) <=100 ))
)