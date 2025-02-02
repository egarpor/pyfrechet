from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricLogEuclidean
from .riemannian_manifold import RiemannianManifold

class LogEuclidean(RiemannianManifold):
    """
    Log-Euclidean Riemannian metric space for nxn SPD matrices.
    
    The distance between two SPD matrices A and B is defined as:
    d(A, B) = || log(A) - log(B) ||_F
    
    .. [1] Gallier, J., Quaintance, J. (2020). `The Log-Euclidean Framework Applied to SPD Matrices`. 
        In: Differential Geometry and Lie Groups. Geometry and Computing, vol 12. Springer, Cham. https://doi.org/10.1007/978-3-030-46040-2_22
    """
    def __init__(self, dim):
        super().__init__(SPDMatrices(n = dim, metric = SPDMetricLogEuclidean(n = dim)))

    def __str__(self):
        return f'SPD_matrices (log-Euclidean metric) (dim={self.manifold.n})'
