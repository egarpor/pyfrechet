import numpy as np
from pyfrechet.metric_spaces import MetricSpace 
from scipy.linalg import eigvals, logm
from geomstats.geometry.spd_matrices import SPDMatrices, SPDLogEuclideanMetric
from geomstats.learning.frechet_mean import FrechetMean
from pyfrechet.metric_spaces.utils import vectorize, devectorize
# from pyriemann.utils.distance import distance_logeuclid

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
        self.manifold = SPDMatrices(n = dim)
        self.manifold.metric = SPDLogEuclideanMetric(space=self.manifold)

    def _check_inputs(self, A, B):
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            raise ValueError("Inputs must be ndarrays")
        if not A.shape == B.shape:
            raise ValueError("Inputs must have equal dimensions")
        if A.ndim < 2:
            raise ValueError("Inputs must be at least a 2D ndarray")

    def _d(self, v1, v2):
        """
        Computes the log-Euclidean distance between two SPD matrices S1 and S2.
        Implementation from pyriemann.utils.distance. See 
        https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/distance.py

        Parameters:
            S1 (ndarray): Symmetric positive definite matrix of shape dxd.
            S2 (ndarray): Symmetric positive definite matrix of shape dxd.

        Returns:
            float: The log-Euclidean distance between S1 and S2.
        """
        
        # Note: manifold.dist(S1, S2) returns the same output, but is slower. Although we still use
        # geomstats for the Fréchet mean, the medoids only require computing the distances, so using this function saves time.
        S1 = devectorize(v1)
        S2 = devectorize(v2)
        self._check_inputs(S1, S2)
        
        return np.linalg.norm(logm(S1) - logm(S2), ord='fro', axis=(-2, -1))

    def _frechet_mean(self, vectors, w):
        matrices = np.array([devectorize(v) for v in vectors])
        mean = FrechetMean(metric=self.manifold.metric, point_type='matrix', verbose=False)
        mean.fit(matrices, weights=w)
        matrix_frechet_mean = mean.estimate_
        return vectorize(matrix_frechet_mean)

    def __str__(self):
        return f'CustomLogEuclidean({self.dim}x{self.dim})'