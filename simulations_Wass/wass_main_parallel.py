import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
sys.path.append(os.getcwd())
import pickle 
import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 # Set the correct path to the pyfrechet module
#sys.path.insert(1, 'C:/Users/Diego/Desktop/Doctorado/codi/pballs')
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metric_spaces import MetricData, Wasserstein1D
from pyfrechet.metric_spaces import wasserstein_1d as ws
from scipy import stats 

n_samples=len(os.listdir(os.path.join(os.path.join(os.getcwd(), 'simulations_Wass'), 'wass_data/')))

def task(file) -> None:
    # Data from the selected file
    sign_level = np.array([0.01, 0.05, 0.1])
    with open(os.path.join(os.path.join(os.getcwd(), 'simulations_Wass'), 'wass_data/' + file), 'rb') as f:
        sample = pickle.load(f)
        
    X=np.c_[sample['x']]

    y = np.array(sample['y'])

    base = Tree(split_type='2means',
                impurity_method='cart',
                mtry=None,
                min_split_size=1)
    
    forest = BaggedRegressor(estimator=base,
                                n_estimators = 200,
                                bootstrap_fraction=1,
                                bootstrap_replace=True,
                                n_jobs=1)
    

    M = Wasserstein1D()
    y = MetricData(M, y)
    
    forest.fit(X, y)
    
    
    results = { 'x_train_data': X,
                'y_train_data': y.data,
                'train_predictions': forest.predict(X).data,
                'forest': forest,
                }
    

    filename = file[:-4] + '_results'
    np.save(filename, results)


Parallel(n_jobs=-1, verbose=40)(
    delayed(task)(file)
    for file in os.listdir(os.path.join(os.getcwd(), 'simulations_Wass', 'wass_data/'))
    if (file.endswith('.pkl') and not os.path.exists(os.path.join(os.getcwd(), 'simulations_Wass', 'wass_results/' + 'WASS_Samp' +  file[9:-4]+ '_results.npy' )))
)