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
import multiprocessing

# By-blocks execution
# n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data')))
# n_cores=8
# #n_cores=int(input('Introduce number of cores: '))
# n_blocks = n_samples/n_cores
# current_block = int(sys.argv[1])

M = Euclidean(dim=1)
# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 3, 5]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

base = Tree(split_type='2means', mtry=2, impurity_method='cart')
forest = BaggedRegressor(estimator=base, n_estimators=100, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=1)
    
def tune_forest(X, y, forest=forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """

    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=1)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_


def task(file) -> None:    
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_euc', 'data', file), 'rb') as f:
        sample = pickle.load(f)
    
    # Define metric space
    X = sample['X']
    y = MetricData(M, sample['Y'])
    
    # Split into train/test sets (50/50 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    # scaler=MinMaxScaler(feature_range=(0,1))
    # X_train=scaler.fit_transform(X_train)
    # X_test=scaler.transform(X_test)

    # Measure time for tuning and fitting
    start_time = time.time()

    # Perform hyperparameter tuning
    best_forest = tune_forest(X_train, y_train, forest, param_grid)
    
    # Fit the best forest
    best_forest.fit(X_train, y_train)

    end_time = time.time()
    tuning_fitting_time = end_time - start_time

    # Store results
    results = {
        'x_test_data': X_test,
        'y_test_data': y_test,
        'forest': best_forest,
        'tuning_fitting_time': tuning_fitting_time
    }
    
    filename = os.path.join(os.getcwd(), 'simulations_euc', 'conformal_results', file[:-4] + '_results.npy')
    np.save(filename, results)

# if __name__ == '__main__':
#     with multiprocessing.Pool() as pool:
#         pool.map(task, os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data/')))



Parallel(n_jobs=8, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data/'))
    if (file.endswith('.pkl')) and not os.path.exists(os.path.join(os.getcwd(), 'simulations_euc', 'conformal_results/' + 'euc_samp' +  file[8:-4]+ '_results.npy' ))
)