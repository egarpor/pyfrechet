import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle 
import numpy as np

from pyfrechet.metrics import mse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed

from pyfrechet.metric_spaces import MetricData, CustomLogEuclidean, CustomAffineInvariant, LogCholesky, spd_to_log_chol
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree


# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data')))
n_cores=56
#n_cores=int(input('Introduce number of cores: '))
n_blocks = n_samples/n_cores
current_block = int(sys.argv[1])

# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 5]
}

# Custom scorer (negative mean squared error, assuming mse is defined elsewhere)
# neg_mse = make_scorer(mse, greater_is_better=False)

# def tune_forest(X, y):
#     """ Perform hyperparameter tuning using GridSearchCV. """
#     base = Tree(split_type='2means', mtry = None, impurity_method='cart')
#     forest = BaggedRegressor(estimator=base, n_estimators=100, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=1)
#     
#     tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=1, verbose=4)
#     tuned_forest.fit(X, y)
#     return tuned_forest.best_estimator_


def task(file) -> None:
    # Data from the selected file
    with open(os.path.join(os.getcwd(), 'simulations_SPD', 'data/' + file), 'rb') as f:
        sample = pickle.load(f)
    X=np.c_[sample['t']]

    sample_Y = np.array(sample['y'])

    base = Tree(split_type='2means',
                impurity_method='cart',
                mtry=None, # It is a regression curve setting, only one predictor
                min_split_size=1)
    
    forest = BaggedRegressor(estimator=base,
                                n_estimators=100,
                                bootstrap_fraction=1,
                                bootstrap_replace=True,
                                n_jobs=1)
    
    for dist in ['LC', 'AI', 'LE']:
        if dist == 'LC':
            M=LogCholesky(dim=2)
            sampleY_LogChol = np.c_[[spd_to_log_chol(A) for A in sample['y']]]
            y = MetricData(M, sampleY_LogChol)
            # Perform hyperparameter tuning
            forest.fit(X, y)

            results = { 'x_train_data': X,
                        'y_train_data': y.data,
                        'train_predictions': forest.predict(X).data,
                        'forest': forest,
                        }
            
        elif dist == 'AI':
            M=CustomAffineInvariant(dim=2)
            y=MetricData(M, sample_Y)

            # Perform hyperparameter tuning
            forest.fit(X, y)

            results = { 'x_train_data': X,
                        'y_train_data': y.data,
                        'train_predictions': forest.predict_matrix(X).data,
                        'forest': forest,
                        }
            
        elif dist == 'LE':
            M=CustomLogEuclidean(dim=2)
            y=MetricData(M, sample_Y)

            results = { 'x_train_data': X,
            'y_train_data': y.data,
            'train_predictions': forest.predict_matrix(X).data,
            'forest': forest,
            }  

        filename = 'simulations_SPD/results/' + dist + '_' + file[:-4] + '_block_' + str(current_block) + '_results'
        np.save(filename, results)


print(f'Block number: {current_block}')
# One sample by core in the current block

Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_SPD', 'data/'))[n_cores*(current_block-1):n_cores*(current_block)]
    if (file.endswith('.pkl')) #and not os.path.exists(os.path.join(os.getcwd(), 'simulations_SPD', 'results/' + 'WASS_Samp' +  file[9:-4]+ '_results.npy' )))
)