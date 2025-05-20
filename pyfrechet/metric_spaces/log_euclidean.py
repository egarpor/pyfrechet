from geomstats.geometry.spd_matrices import SPDMatrices, SPDLogEuclideanMetric
from geomstats.learning.frechet_mean import FrechetMean
from pyfrechet.metric_spaces.utils import vectorize, devectorize
from .riemannian_manifold import RiemannianManifold
import numpy as np

class LogEuclidean(RiemannianManifold):
    """
    Log-Euclidean Riemannian metric space for nxn SPD matrices.
    Wraps SPD matrices for sklearn compatibility with scikit-learn by vectorizing them.

    
    The distance between two SPD matrices A and B is defined as:
    d(A, B) = || log(A) - log(B) ||_F
    
    .. [1] Gallier, J., Quaintance, J. (2020). `The Log-Euclidean Framework Applied to SPD Matrices`. 
        In: Differential Geometry and Lie Groups. Geometry and Computing, vol 12. Springer, Cham. https://doi.org/10.1007/978-3-030-46040-2_22
    """
    def __init__(self, dim):
        super().__init__(SPDMatrices(n = dim, metric = SPDMetricLogEuclidean(n = dim)))

    def __str__(self):
        return f'SPD_matrices (log-Euclidean metric) (dim={self.manifold.n})'

    def _d(self, v1, v2):
        """
        Computes the log-Euclidean distance between two SPD matrices S1 and S2.

        Parameters:
            S1 (ndarray): Symmetric positive definite matrix of shape dxd.
            S2 (ndarray): Symmetric positive definite matrix of shape dxd.

        Returns:
            float: The log-Euclidean distance between S1 and S2.
        """
        S1 = devectorize(v1)
        S2 = devectorize(v2)
        return self.manifold.metric.dist(S1, S2)

    def _frechet_mean(self, vectors, w):
        matrices = np.array([devectorize(v) for v in vectors])
        mean = FrechetMean(metric=self.manifold.metric, point_type='matrix', verbose=False)
        mean.fit(matrices, weights=w)
        matrix_frechet_mean = mean.estimate_
        return vectorize(matrix_frechet_mean)
