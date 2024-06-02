import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import get_cmap

# from ipywidgets import *

from typing import Union

import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
import datetime as dt
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.linalg import expm, logm
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wishart
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn import neighbors, clone


from pyfrechet.metric_spaces import MetricData, Euclidean, LogCholesky, spd_to_log_chol, log_chol_to_spd
# from pyfrechet.regression.frechet_regression import LocalFrechet, GlobalFrechet
# from pyfrechet.regression.kernels import NadarayaWatson, gaussian, epanechnikov
# from pyfrechet.regression.knn import KNearestNeighbours
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

with open('taxi_data_jan_joined.pkl', 'rb') as f:
    df_jan = pickle.load(f)

with open('taxi_data_feb_joined.pkl', 'rb') as f:
    df_feb = pickle.load(f)

with open('taxi_data_jan_Y.pkl', 'rb') as f:
    Y_jan=pickle.load(f)

with open('taxi_data_feb_Y.pkl', 'rb') as f:
    Y_feb=pickle.load(f)

Y= Y_jan + Y_feb
df=pd.concat([df_jan, df_feb], axis=0, ignore_index=True)

# Without train-train and train-validation splitting
M=LogCholesky(dim=10)

X=df[['Temp.Avg', 'DewPoint.Avg', 'Humidity.Avg', 'WindSpeed.Avg', 
          'Pressure.Avg', 'Precipitation', 'Hour.Indicator', 'Weekday']]
X=pd.get_dummies(data=X, columns=['Hour.Indicator', 'Weekday'])
Y_stand=[A/A.max() for A in Y] # Standardize the weights
Y_ExpLogChol=np.c_[[spd_to_log_chol(expm(A)) for A in Y_stand]]
y=MetricData(M, Y_ExpLogChol)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=283, random_state=100)
scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

param_grid={
    'estimator__min_split_size': [1, 5, 10, 15, 20, 25, 30],
    'estimator__mtry': [1, 2, 3, 4, 5, 6, 8, 10, 12]
}
# param_grid={
#     'estimator__min_split_size': [1, 5],
#     'estimator__mtry': [1, 4]
# }


# In SearchGridCV we must use a score (greater the better) rather than a loss function (mse)
neg_mse=make_scorer(score_func=mse, 
                    greater_is_better=False,
                    sample_weight=None) # Since mse is a loss function, not a score function

# If cv is not specified, 5-CV is used by default
tuned_forest=GridSearchCV(
    estimator=BaggedRegressor(estimator=Tree(split_type='2means',
                                             impurity_method='medoid'),
                            n_estimators=100,
                            bootstrap_fraction=1,
                            bootstrap_replace=True),
    param_grid=param_grid,
    scoring=neg_mse,
    cv=5,
    n_jobs=-1,
    verbose=4
)

start_time=time.time()
tuned_forest.fit(X_train, y_train)
end_time=time.time()
print(f'Tuning execution time: {end_time-start_time}')

joblib.dump(tuned_forest, 'NY_tuned_forest_1.joblib')
