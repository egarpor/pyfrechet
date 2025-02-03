from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pyfrechet.metric_spaces import MetricData
from pyfrechet.metrics import r2_score

# Create T as the type variable constained to be a subtype of WeightingRegressor class
T = TypeVar("T", bound="WeightingRegressor")

class WeightingRegressor(RegressorMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, precompute_distances: bool=False):
        self.precompute_distances = precompute_distances

    def _normalize_weights(self, weights: np.ndarray, 
                           sum_to_one: bool=False, 
                           clip: bool=False, 
                           clip_allow_neg: bool=False) -> np.ndarray:
        """
        Normalization of weights.

        clip indicates whether to set as 0 weights lower than machine precision eps
        (for more computational efficiency).
        clip_allow_neg indicates if negative weights are allowed (in addition to values close to eps
        being set to 0)
        sum_to_one controls if normalization of weights must be done to sum 1
        """
        if sum_to_one:
            weights /= weights.sum()
        
        if clip:
            # Get the machine limits of floating point representation for the dtype of weigths
            eps = np.finfo(weights.dtype).eps

            if clip_allow_neg:
                # Setting a_max=None means no clipping in that edge
                clipped = np.clip(np.abs(weights), a_min=eps, a_max=None)
                weights[clipped == eps] = 0.0
            else:
                # Setting a_max=None means no clipping in that edge
                # (Note there is no np.abs() here, as we do not allow negative weights)
                weights = np.clip(weights, a_min=eps, a_max=None)
                weights[weights == eps] = 0.0
        
            if sum_to_one:
                weights /= weights.sum()
        
        return weights

    def _predict_one(self, x):        
        """
        Make prediction for just one new observation (requirement for predict() method).

        The prediction is computed as the Frechet mean with weights given by the estimator itself.
        """
        return self.y_train_.frechet_mean(self.weights_for(x))
    
    def _oob_predict_one(self, x):        
        """
        Make OOB prediction for just one new observation (requirement for oob_predict() method).
        """
        return self.y_train_.frechet_mean(self.oob_weights_for(x))
    
    @abstractmethod
    def fit(self:T, X, y: MetricData) -> T:
        self.X_train_ = X
        if y.data.ndim ==1:
            # y cannot have np.nan or np.inf when multi_output=True
            # There is also a way of allowing np.nan (see sklearn help)
            X, _ = check_X_y(X, y.data, multi_output=True, allow_nd=True)
            
        else:
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("X contains NaNs or infinite values")
            if np.any(np.isnan(y.data)) or np.any(np.isinf(y.data)):
                raise ValueError("y contains NaNs or infinite values")
            
        self.y_train_ = y

        if self.precompute_distances:
            y.compute_distances()

        return self
    
    @abstractmethod
    def weights_for(self, x) -> np.ndarray:
        pass



    def predict(self, x):
        # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
        check_is_fitted(self)
        x = check_array(x) 

        # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._predict_one(x)
        else:
            y0 = self._predict_one(x[0,:])
            # The shape of predictions will have x.shape[0] (number of observations for prediction) rows
            # and y0.shape[0] (length (dimension) of the MetricData class we are handling) columns
            y_pred = np.zeros((x.shape[0], y0.shape[0]))
            y_pred[0,:] = y0
            for i in range(1, x.shape[0]):
                y_pred[i,:] = self._predict_one(x[i,:])
            return MetricData(self.y_train_.M, y_pred)
        
    def predict_matrix(self, x):
        #TO DO: CHECK that AN ARRAY IS PASSED? UNEXPECTED BEHAVIOUR IF AN ARRAY IS NOT PASSED AND WE HAVE A MATRIX INSTEAD
        # If y is a matrix:
        # Check if estimator has been fitted (.fit() has been applied), otherwise raise an error
        check_is_fitted(self)
        x = check_array(x)

        # To allow (1,p) and (,p) shapes ('matrices' and 'vectors')
        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._predict_one(x)
        else:
            y0 = self._predict_one(x[0,:])
            
            y_pred = np.zeros((x.shape[0], y0.shape[0], y0.shape[1]))
            y_pred[0,:,:] = y0
            for i in range(1, x.shape[0]):
                y_pred[i,:,:] = self._predict_one(x[i,:])
            return MetricData(self.y_train_.M, y_pred)
    
    def score(self, X, y: MetricData, sample_weight=None, force_finite=True):
        """Return the determination coefficient R^2 between target and fitted values.
        
        force_finite=True replace np.nan and np.inf scores resulting from constant data 
        by 1 if prediction is perfect and 0 otherwise (True is convenient).
        """
        return r2_score(y, self.predict(X), sample_weight=sample_weight, force_finite=force_finite)