import numpy as np
from pyfrechet.metric_spaces import MetricSpace 
from scipy.linalg import eigvals
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricLogEuclidean
from geomstats.learning.frechet_mean import FrechetMean
from pyriemann.utils.distance import distance_logeuclid

class CustomLogEuclidean(MetricSpace):
    """
    Log-Euclidean Riemannian metric space for nxn SPD matrices.
    
    The distance between two SPD matrices A and B is defined as:
    d(A, B) = || log(A) - log(B) ||_F
    
    .. [1] Gallier, J., Quaintance, J. (2020). `The Log-Euclidean Framework Applied to SPD Matrices`. 
        In: Differential Geometry and Lie Groups. Geometry and Computing, vol 12. Springer, Cham. https://doi.org/10.1007/978-3-030-46040-2_22
    """
    def __init__(self, dim):
        self.dim = dim 
        self.manifold = SPDMatrices(n = dim, metric = SPDMetricLogEuclidean(n = dim))

    def _d(self, S1, S2):
        """
        Computes the log-Euclidean distance between two SPD matrices S1 and S2.

        Parameters:
            S1 (ndarray): Symmetric positive definite matrix of shape dxd.
            S2 (ndarray): Symmetric positive definite matrix of shape dxd.

        Returns:
            float: The log-Euclidean distance between S1 and S2.
        """
        # Note: manifold.dist(S1, S2) returns the same output, but is slower. Although we still use
        # geomstats for the Fréchet mean, the medoids only require computing the distances, so using this function saves time.
        return distance_logeuclid(S1, S2)


    def _frechet_mean(self, y, w):
        mean = FrechetMean(metric=self.manifold.metric)
        mean.fit(y, weights=w)
        return mean.estimate_

    def __str__(self):
        return f'CustomLogEuclidean({self.dim}x{self.dim})'
