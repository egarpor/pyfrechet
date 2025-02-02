import sys, os
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle 
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from scipy.linalg import expm
from scipy.stats import wishart
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 # Set the correct path to the pyfrechet module
#sys.path.insert(1, 'C:/Users/Diego/Desktop/Doctorado/codi/pballs')
from pyfrechet.metric_spaces import MetricData, LogCholesky, spd_to_log_chol, log_chol_to_spd
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree

M=LogCholesky(dim=2)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data')))
print(os.path.join(os.getcwd(), 'simulations_SPD', 'data'))
n_cores=56
#n_cores=int(input('Introduce number of cores: '))
n_blocks = n_samples/n_cores
current_block = 4

def task(file) -> None:
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_SPD', 'data\\' + file), 'rb') as f:
        sample = pickle.load(f)
    X=np.c_[sample['sample'][0]]
    sampleY_LogChol=np.c_[[spd_to_log_chol(A) for A in sample['sample'][1]]]
    y=MetricData(M, sampleY_LogChol)

    # Train/test partition and scaling data
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=100)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base = Tree(split_type='2means',
            impurity_method='cart',
            mtry=None, # It is a regression curve setting, only one predictor
            min_split_size=1)
    forest = BaggedRegressor(estimator=base,
                            n_estimators=100,
                            bootstrap_fraction=1,
                            bootstrap_replace=True,
                            n_jobs=1)
    forest.fit(X_train, y_train)
    
    results = {'train_indices': train_idx.tolist(),
               'y_train_data': y_train.data.tolist(),
               'train_predictions': forest.predict(X_train).data.tolist(),
               'y_test_data': y_test.data.tolist(),
               'test_predictions': forest.predict(X_test).data.tolist(),
               'oob_errors': forest.oob_errors().tolist()}

    filename = 'LC_' + file[:-4] + '_block_' + str(current_block) + '_results.json'
    with open(os.path.join(os.getcwd(), 'simulations_SPD', 'results\\' + filename), 'w', encoding = 'utf8') as f:
        json.dump(results, f)


print(f'Block number: {current_block}')
# One sample by core in the current block

Parallel(n_jobs=-1, verbose=40)(delayed(task)(file) for file in \
        os.listdir(os.path.join(os.path.dirname(__file__), 'data'))[n_cores*(current_block-1):n_cores*(current_block)])   