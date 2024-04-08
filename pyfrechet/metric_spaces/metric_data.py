import numpy as np
from typing import Union, Any, TypeVar
import numbers
from .utils import D_mat_par, D_mat, mat_sel_idx, mat_sel, coalesce_weights
from .metric_space import MetricSpace

# Create T as the type variable constained to be a subtype of MetricData class
T = TypeVar("T", bound="MetricData")

class MetricData:
    def __init__(self, M: MetricSpace, data, distances=None):
        self.M = M
        self.data = data
        self.distances = distances
        self.shape = (data.shape[0],) # data.shape[0] is the sample size

    def compute_distances(self, n_jobs: Union[int, None] =-2):
        if self.distances is None:
            if n_jobs is None or n_jobs == 1:
                self.distances = D_mat(self.M, self.data)
            else:
                self.distances = D_mat_par(self.M, self.data, n_jobs)

    def frechet_mean(self, weights: Union[np.ndarray, None] =None):
        return self.M.frechet_mean(self.data, weights)

    def frechet_var(self, weights: Union[np.ndarray, None] =None):
        return self.M.frechet_var(self.data, weights)
    
    def frechet_medoid(self, weights: Union[np.ndarray, None] =None, n_jobs: Union[int, None] =-2):
        """
        Compute the medoid for the weighted Frechet mean. That is, gets the data point
        within the sample that minimizes the weighted sum of distances with the rest of
        individuals of the sample.
        """
        self.compute_distances(n_jobs=n_jobs)
        weights = coalesce_weights(weights, self)
        idx = np.argmin(self.distances.dot(weights))
        return self[idx]

    def frechet_medoid_var(self, weights: Union[np.ndarray, None] =None, n_jobs: Union[int, None] =-2):
        """
        Compute the medoid weighted Frechet variance. 
        In other words, the target minimum value obtained from frechet_medoid() optimization problem.
        """
        self.compute_distances(n_jobs=n_jobs)
        weights = coalesce_weights(weights, self)
        return np.min(self.distances.dot(weights))

    def __getitem__(self, key) -> Union[Any, T]:
        """
        (Magic method) Enables the objects to behave like containers.
        
        It is automatically called when using indexer brackets [].
        """
        subset = self.M.index(self.data, key)
        if isinstance(key, numbers.Integral):
            return subset
        elif self.distances is None:
            return MetricData(self.M, subset)
        else:
            key = key if type(key) is np.ndarray else np.array(key)
            subdist = mat_sel(self.distances, key) if key.dtype == 'bool' else mat_sel_idx(self.distances, key)
            return MetricData(self.M, subset, subdist)
        
    def __len__(self):
        """(Magic method) Returns the length of the MetricData object (sample size)"""
        return self.data.shape[0]
    
    def __str__(self):
        """(Magic method) Returns text information about the object. It is called by print() function."""
        return f'MetricData(M={self.M}, len={len(self)}, has_distance={not self.distances is None})'
    

class MetricBall:
    def __init__(self, M: MetricSpace, center: np.ndarray, radius: float):
        self.M = M
        self.center = center
        self.radius = radius

    def isin_Ball(self, x: np.ndarray) -> np.ndarray:
        return self.M.d(self.center, x) < self.radius