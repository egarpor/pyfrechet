from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional, Union
import sklearn

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor
from sklearn.utils.validation import check_array, check_is_fitted
import warnings


class BaggedRegressor(WeightingRegressor):
    def __init__(self, 
                 estimator: Optional[WeightingRegressor]=None,
                 n_estimators: int=100,
                 bootstrap_fraction: float=0.75,
                 bootstrap_replace: bool=False,
                 n_jobs: Optional[int]=-2,
                 verbose: int=0):
        """
        .estimator is the base learner for the BaggedRegressor (e.g. Tree)
        .estimators is a list composed of as many tuples (subsample, estimator trained with subsample)
        as n_estimators.
        """
        super().__init__()
        self.estimator = estimator
        self.precompute_distances = estimator.precompute_distances if estimator else False
        self.n_estimators = n_estimators
        self.estimators: list[tuple[np.ndarray, WeightingRegressor]] = []
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_replace = bootstrap_replace
        self.n_jobs = n_jobs
        self.verbose=verbose

    def _make_mask(self, N: int) -> np.ndarray:
        """
        Method for randomly select the training subsample of each base learner.

        By default it takes bootstrap samples without replacement of size bootstrap_fraction * N.
        If bootstrap_fraction=1 and bootstrap_replace=True, the usual bootstrap approach is performed.
        """
        s = int(self.bootstrap_fraction * N)
        return np.random.choice(N, size=s, replace=True) if self.bootstrap_replace \
            else np.random.choice(N, size=s, replace=False)

    def _fit_est(self, X, y: MetricData) -> tuple[np.ndarray, object]:
        """Method to train a single base estimator of the ensemble.
        
        It returns a tuple with the mask that identifies the subsample employed and the 
        fitted estimator."""
        mask = self._make_mask(X.shape[0])
        # Clone does a deep copy of the model in an estimator without actually copying attached data
        # It yields a new estimator with the same parameters that has not been fit on any data
        return (mask, sklearn.clone(self._estimator).fit(X[mask, :], y[mask]))

    def _fit_par(self, X, y: MetricData):
        """
        Method to train all the estimators in the ensemble using parallel computing.
        """
        super().fit(X, y)
        def calc(): return self._fit_est(X, y)
        self.estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(calc)() 
                                                                                      for _ in range(self.n_estimators)) or []
        return self

    def _fit_seq(self, X, y: MetricData):
        """
        Method to train all the estimators in the ensemble using non-parallel computing (sequential).
        """
        super().fit(X, y)
        if self.verbose==0:
            self.estimators = [ self._fit_est(X, y) for _ in range(self.n_estimators)]
        else:
            self.estimators = [ self._fit_est(X, y) for _ in tqdm(range(self.n_estimators))]
        return self

    # WeightingRegressor has 2 abstract methods: fit() and weights_for()
    def fit(self, X, y: MetricData):
        super().fit(X, y)
        
        assert self.estimator, "No base estimator has been provided to BaggedRegressor"
        self._estimator = self.estimator
        
        return self._fit_seq(X, y) if self.n_jobs == 1 or not self.n_jobs else self._fit_par(X, y)
    
    # Weights for OOB predictions
    def oob_weights_for(self, x) -> np.ndarray:
        count = 0
        assert len(self.estimators) > 0, "At least one estimator is needed to compute weights"
        # Index of the observation x in the training data
        if self.X_train_.shape[1] == 1:
            x_idx = np.argwhere(np.isclose(self.X_train_, x, rtol=1e-10).flatten()).flatten()
        else:
            x_idx = np.argwhere(np.all(np.isclose(self.X_train_, x, rtol=1e-10), axis=1)).flatten()
        if len(x_idx) == 0:
            x_idx = None
            warnings.warn('Observation not found in training data, working with the full training set')
        else:
            x_idx = x_idx[0]
        weights = np.zeros(self.y_train_.shape[0])
        for (mask, estimator) in self.estimators:
            # Weights for non-OOB observations are zero, so that they do not have an effect over the prediction
            if x_idx in mask:
                est_weights = np.repeat(0, self.y_train_.shape[0])
            else:
                # Note that each base estimator has its own .weights_for() method
                est_weights = estimator.weights_for(x)
                #Count how many trees there are for which x is OOB. Notice that now, we aggregate only the weights of these trees, not over all the trees. 
                count += 1
            weights[mask] += est_weights
        return self._normalize_weights(weights / count, clip=True)

    def weights_for(self, x) -> np.ndarray:
        assert len(self.estimators) > 0, "At least one estimator is needed to compute weights"
        weights = np.zeros(self.y_train_.shape[0])
        for (mask, estimator) in self.estimators:
            # Note that each base estimator has its own .weights_for() method
            est_weights = estimator.weights_for(x)
            weights[mask] += est_weights
        return self._normalize_weights(weights / self.n_estimators, clip=True)
    
    def _oob_predict_one(self, x):        
        """
        Make OOB prediction for just one new observation (requirement for oob_predict() method).
        """
        return self.y_train_.frechet_mean(self.oob_weights_for(x))
    
    def oob_predict(self, x):
        # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
        check_is_fitted(self)
        x = check_array(x) 

        # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._oob_predict_one(x)
        else:
            y0 = self._oob_predict_one(x[0,:])
            # The shape of predictions will have x.shape[0] (number of observations for prediction) rows
            # and y0.shape[0] (length (dimension) of the MetricData class we are handling) columns
            y_pred = np.zeros((x.shape[0], y0.shape[0]))
            y_pred[0,:] = y0
            for i in range(1, x.shape[0]):
                y_pred[i,:] = self._oob_predict_one(x[i,:])
            return MetricData(self.y_train_.M, y_pred)
    
    def oob_errors(self) -> np.ndarray:
        """
        Compute a np.ndarray with the OOB prediction errors for each of the training observations.
        Each row corresponds to one observation.

        Note that each error is the distance (inherited from the MetricSpace underlying object)
        between the training observation target and its OOB prediction.
        """
        oob_preds=self.oob_predict(self.X_train_)
        return oob_preds.M.d(self.y_train_.data, oob_preds.data)

    def oob_predict_matrix(self, x: np.ndarray) -> MetricData:
        """
        Matrix-valued response
        """
        # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
        check_is_fitted(self)

         # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._oob_predict_one(x)
        else:
            y0 = self._oob_predict_one(x[0,:])
            # The shape of predictions will have x.shape[0] (number of observations for prediction) rows
            # and y0.shape[0] (length (dimension) of the MetricData class we are handling) columns
            oob_pred = np.zeros((x.shape[0], y0.shape[0], y0.shape[1]))
            oob_pred[0,:,:] = y0
            for i in range(1, x.shape[0]):

                oob_pred[i,:,:] = self._oob_predict_one(x[i,:])
                
            return MetricData(self.y_train_.M, oob_pred)

    def oob_errors_matrix(self) -> np.ndarray:
        """
        OOB errors for matrix valued response
        """
        oob_preds=self.oob_predict_matrix(self.X_train_)
        return oob_preds.M.d(self.y_train_, oob_preds)
    
    #def _oob_predict_one(self, x: np.ndarray) -> np.ndarray:
    #    """
    #    Predicts observation x using only the estimators in which x is OOB (out-of-bag).
    #    (INTERNAL USE ONLY)
    #    
    #    This function will be used to compute OOB prediction errors.
    #    x is assumed to be a member of the training sample, otherwise it would be OOB in all the ensemble.
    #    """
    #    assert len(self.estimators) > 0, "At least one estimator is needed to compute OOB predictions"
#
    #    # Index of the observation x in the training data
    #    x_idx = np.argwhere(np.all(self.X_train_ == x, axis = 1)).flatten()[0]
#
    #    # Estimators of the ensemble in which x is OOB
    #    oob_estimators_idx = [idx for idx in range(self.n_estimators) if x_idx not in self.estimators[idx][0]]
#
    #    try:
    #        y0 = self.estimators[oob_estimators_idx[0]][1].predict(x.reshape(1,-1)).data  
    #        oob_preds = np.zeros((len(oob_estimators_idx), y0.shape[0]))
    #        oob_preds[0,:] = y0
    #    except IndexError:
    #        # Return the Frechet mean of the training data
    #        return MetricData(self.y_train_.M, self.y_train_.data).frechet_mean()
    #    
#
    #    for i in range(1, len(oob_estimators_idx[1:]) + 1):
    #        # Each row contains the prediction of the i-th tree for the point x
    #        oob_preds[i,:] = self.estimators[oob_estimators_idx[i]][1].predict(x.reshape(1,-1)).data
#
    #    return MetricData(self.y_train_.M, oob_preds).frechet_mean()
    #
    #def _oob_predict_one_matrix(self, x: np.ndarray) -> np.ndarray:
    #    """
    #    Predicts observation x using only the estimators in which x is OOB (out-of-bag).
    #    (INTERNAL USE ONLY)
    #    
    #    This function will be used to compute OOB prediction errors.
    #    x is assumed to be a member of the training sample, otherwise it would be OOB in all the ensemble.
    #    """
    #    assert len(self.estimators) > 0, "At least one estimator is needed to compute OOB predictions"
#
    #    # Index of the observation x in the training data
    #    x_idx = np.argwhere(np.all(self.X_train_ == x, axis = 1)).flatten()[0]
#
    #    # Estimators of the ensemble in which x is OOB
    #    oob_estimators_idx = [idx for idx in range(self.n_estimators) if x_idx not in self.estimators[idx][0]]
#
    #    try:
    #        y0 = self.estimators[oob_estimators_idx[0]][1].predict_matrix(x.reshape(1,-1)).data  
    #        oob_preds = np.zeros((len(oob_estimators_idx), y0.shape[0], y0.shape[1]))
    #        oob_preds[0,:,:] = y0
    #    except IndexError:
    #        # Return the Frechet mean of the training data
    #        return MetricData(self.y_train_.M, self.y_train_.data).frechet_mean()
    #    
    #    for i in range(1, len(oob_estimators_idx[1:]) + 1):
    #        # Each row contains the prediction of the i-th tree for the point x
    #        oob_preds[i,:,:] = self.estimators[oob_estimators_idx[i]][1].predict_matrix(x.reshape(1,-1)).data
#
    #    return MetricData(self.y_train_.M, oob_preds).frechet_mean()
    #
    #def oob_predict(self, x: np.ndarray) -> MetricData:
    #    """
    #    Obtain a MetricData object with the OOB predictions of the ensemble for argument x. 
    #    """
    #    # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
    #    check_is_fitted(self)
#
    #     # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
    #    if len(x.shape) == 1 or x.shape[0] == 1:
    #        return self._oob_predict_one(x)
    #    else:
    #        y0 = self._oob_predict_one(x[0,:])
    #        # The shape of predictions will have x.shape[0] (number of observations for prediction) rows
    #        # and y0.shape[0] (length (dimension) of the MetricData class we are handling) columns
    #        oob_pred = np.zeros((x.shape[0], y0.shape[0]))
    #        oob_pred[0,:] = y0
    #        for i in range(1, x.shape[0]):
    #            oob_pred[i,:] = self._oob_predict_one(x[i,:])
    #            
    #        return MetricData(self.y_train_.M, oob_pred)
    #    
    #def oob_predict_matrix(self, x: np.ndarray) -> MetricData:
    #    """
    #    Matrix-valued response
    #    """
    #    # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
    #    check_is_fitted(self)
#
    #     # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
    #    if len(x.shape) == 1 or x.shape[0] == 1:
    #        return self._oob_predict_one_matrix(x)
    #    else:
    #        y0 = self._oob_predict_one_matrix(x[0,:])
    #        # The shape of predictions will have x.shape[0] (number of observations for prediction) rows
    #        # and y0.shape[0] (length (dimension) of the MetricData class we are handling) columns
    #        oob_pred = np.zeros((x.shape[0], y0.shape[0], y0.shape[1]))
    #        oob_pred[0,:,:] = y0
    #        for i in range(1, x.shape[0]):
    #            oob_pred[i,:,:] = self._oob_predict_one_matrix(x[i,:])
    #            
    #        return MetricData(self.y_train_.M, oob_pred)
    #    
    #def oob_errors(self) -> np.ndarray:
    #    """
    #    Compute a np.ndarray with the OOB prediction errors for each of the training observations.
    #    Each row corresponds to one observation.
#
    #    Note that each error is the distance (inherited from the MetricSpace underlying object)
    #    between the training observation target and its OOB prediction.
    #    """
    #    oob_preds=self.oob_predict(self.X_train_)
    #    return oob_preds.M.d(self.y_train_.data, oob_preds.data)
    #
    #def oob_errors_matrix(self) -> np.ndarray:
    #    """
    #    Matrix-valued response
    #    """
    #    oob_preds=self.oob_predict_matrix(self.X_train_)
    #    return oob_preds.M.d(self.y_train_, oob_preds)



        
