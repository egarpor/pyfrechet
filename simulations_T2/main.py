import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import pickle 
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyfrechet.metric_spaces import MetricData, Torus
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree

M=Torus(2)

for file in tqdm(os.listdir(os.path.join(os.getcwd(), 'data')), desc='Progress'):
    # Data from the selected file
    sample=pd.read_csv(os.path.join(os.getcwd(), 'data/'+file))
    X=sample[['t']].values
    y=MetricData(M, sample[['theta.1', 'theta.2']].values)

    # Train/test partition and scaling data
    train_idx, test_idx=train_test_split(np.arange(len(X)), test_size=0.25)
    X_train=X[train_idx]
    X_test=X[test_idx]
    y_train=y[train_idx]
    y_test=y[test_idx]
    scaler=MinMaxScaler(feature_range=(0,1))
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    base = Tree(split_type='2means',
            impurity_method='cart',
            mtry=None,
            min_split_size=5)
    forest = BaggedRegressor(estimator=base,
                            n_estimators=100,
                            bootstrap_fraction=1,
                            bootstrap_replace=True,
                            n_jobs=-2)
    forest.fit(X_train, y_train)
    
    results={'train_indices': train_idx,
             'y_train_data': y_train.data,
             'train_predictions': forest.predict(X_train).data,
             'y_test_data': y_test.data,
             'test_predictions': forest.predict(X_test).data,
             'oob_errors': forest.oob_errors()}

    filename=file[:-4] + '_results.pkl'
    with open(os.path.join(os.getcwd(), 'results/'+filename), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print(file)
    
