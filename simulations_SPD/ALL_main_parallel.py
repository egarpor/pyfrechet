import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle 
import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 # Set the correct path to the pyfrechet module
#sys.path.insert(1, 'C:/Users/Diego/Desktop/Doctorado/codi/pballs')
from pyfrechet.metric_spaces import MetricData, CustomLogEuclidean, CustomAffineInvariant, LogCholesky, spd_to_log_chol
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree



# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data')))
n_cores=56
#n_cores=int(input('Introduce number of cores: '))
n_blocks = n_samples/n_cores
current_block = int(sys.argv[1])

def task(file) -> None:
    # Data from the selected file
    sign_level = np.array([0.01, 0.05, 0.1])
    with open(os.path.join(os.getcwd(), 'simulations_SPD', 'data/' + file), 'rb') as f:
        sample = pickle.load(f)

    X=np.c_[sample['t']]
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)

    y = np.array(sample['y'])

    base = Tree(split_type='2means',
                impurity_method='cart',
                mtry=None, # It is a regression curve setting, only one predictor
                min_split_size=1)
    
    forest = BaggedRegressor(estimator=base,
                                n_estimators=100,
                                bootstrap_fraction=1,
                                bootstrap_replace=True,
                                n_jobs=1)
    
    for dist in ['AI']:
        #if dist == 'LC':
        #    M=LogCholesky(dim=2)
        #    sampleY_LogChol=np.c_[[spd_to_log_chol(A) for A in sample['sample'][1]]]
        #    y=MetricData(M, sampleY_LogChol)
        #    y_train = y[train_idx]
        #    y_test = y[test_idx]
        #    
        #    forest.fit(X_train, y_train)
        #    
        #    Dalpha = np.percentile(forest.oob_errors(), (1-sign_level)*100)
        #    
        #    results = {'train_indices': train_idx,
        #                'y_train_data': y_train.data,
        #                'train_predictions': forest.predict(X_train).data,
        #                'y_test_data': y_test.data,
        #                'test_predictions': forest.predict(X_test).data,
        #                'oob_errors': forest.oob_errors(),
        #                
        #                'coverage': np.mean(np.array([M.d(forest.predict(X_test).data, y_test.data) <= alpha for alpha in Dalpha]).T, axis = 0),
        #                'OOB_quantile' : Dalpha,
        #                'forest': forest,
        #                }
        #    
        #elif dist == 'LE':
        #    M=CustomLogEuclidean(dim=2)
        #    y=MetricData(M, np.array(sample['sample'][1]))
        #    y_train = y[train_idx]
        #    y_test = y[test_idx]
        #    
        #    forest.fit(X_train, y_train)
        #    
        #    Dalpha = np.percentile(forest.oob_errors_matrix(), (1-sign_level)*100)
        #    
        #    results = {'train_indices': train_idx,
        #                'y_train_data': y_train.data,
        #                'train_predictions': forest.predict_matrix(X_train).data,
        #                'y_test_data': y_test.data,
        #                'test_predictions': forest.predict_matrix(X_test).data,
        #                'oob_errors': forest.oob_errors_matrix(),
        #                'coverage': np.mean(np.array([M.d(forest.predict_matrix(X_test).data, y_test.data) <= alpha for alpha in Dalpha]).T, axis = 0),
        #                'OOB_quantile' : Dalpha,
        #                'forest': forest,
        #                }
        #else:
        M=CustomAffineInvariant(dim=2)
        y=MetricData(M, y)
        
        forest.fit(X, y)
        
        oob_errors = forest.oob_errors_matrix()

        Dalpha = np.percentile(oob_errors, (1-sign_level)*100)
        
        results = {'y_train_data': y.data,
                    'train_predictions': forest.predict_matrix(X).data,
                    'OOB_errors': oob_errors,
                    'OOB_quantile' : Dalpha,
                    'forest': forest,
                    }
        

        filename = dist + '_' + file[:-4] + '_block_' + str(current_block) + '_results'
        np.save(filename, results)


print(f'Block number: {current_block}')
# One sample by core in the current block

for file in os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data'))[n_cores*(current_block-1):n_cores*(current_block)]:
    task(file)