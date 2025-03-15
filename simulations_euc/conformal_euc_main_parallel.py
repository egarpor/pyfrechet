import os
import pickle
import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data_dir = os.path.join(os.getcwd(), 'simulations_euc', 'data')
results_dir = os.path.join(os.getcwd(), 'simulations_euc', 'conformal_results')
os.makedirs(results_dir, exist_ok=True)

# Define parameter grid for tuning
param_grid = {
    'min_samples_leaf': [1, 5, 10],
    'max_features': [1, 2, 3]
    }


def tune_forest(X, y, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    base_forest = RandomForestRegressor(n_jobs=1, random_state=1000, n_estimators=2000, oob_score=True)
    grid_search = GridSearchCV(estimator=base_forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=4)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def task(file):    
    # Load data
    with open(os.path.join(data_dir, file), 'rb') as f:
        sample = pickle.load(f)
    
    X = sample['X']
    y = sample['Y']
    y = y.ravel()
    
    # Split into train/test sets (50/50 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    # Measure time for tuning and fitting
    start_time = time.time()
    
    # Perform hyperparameter tuning
    best_forest = tune_forest(X_train, y_train, param_grid)
    
    end_time = time.time()
    tuning_fitting_time = end_time - start_time
    
    # Store results
    results = {
        'x_test_data': X_test,
        'y_test_data': y_test,
        'forest': best_forest,
        'tuning_fitting_time': tuning_fitting_time
    }
    
    filename = os.path.join(results_dir, file[:-4] + '_results.npy')
    np.save(filename, results)

Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(data_dir)
    if (file.endswith('.pkl') and not os.path.exists(os.path.join(os.getcwd(), 'simulations_euc', 'conformal_results/' +  file[:-4]+ '_results.npy' )))
)