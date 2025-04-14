import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.hypersphere import Hypersphere
from .metric_space import MetricSpace
from geomstats.geometry.hypersphere import Hypersphere


#ATTENTION OJO TODO CHANGE TO RIEMANNIANMANIFORLD FOR PREVIOUS DISTANCE
class AnisotropicSphere(MetricSpace):
    def __init__(self, dim):
        #super().__init__(Hypersphere(dim = dim))
        self.dim = dim
        self.extrinsic_dim = dim + 1
        self.manifold = Hypersphere(dim=dim)
        # Riemannian metric is overridden (no structure of Riemannian manifold)  
        self.dist_override = True

    def _d(self, x, y):
        """
        Computes anisotropic distances row-wise between X and Y (both n x 3 arrays).
        
        Parameters:
        - X, Y: numpy arrays of shape (n, 3), each row is a point on the unit sphere
        - lambda_: anisotropy coefficient; more penalty in z-direction

        Returns:
        - distances: numpy array of shape (n,), the anisotropic distances
        """
        lambda_ = 3
        diff = x - y  # shape (n, 3)
        weights = np.hstack([np.ones(self.extrinsic_dim-1), (1 + lambda_)])
        weighted_diff = diff**2 * weights  # broadcast multiplication
        return np.sqrt(np.sum(weighted_diff.reshape(-1, weighted_diff.shape[-1]), axis=1))


    def _frechet_mean(self, y, w=None):
        # Important: it is better to use medoids here!
        extrinsic_mean = w.dot(y)
        return extrinsic_mean / np.linalg.norm(extrinsic_mean)

    def __str__(self):
        return f'Sphere, anisotropic distance (dim={self.manifold.dim})'
 
def r2_to_angle(x):
    return Hypersphere(dim=1).extrinsic_to_angle(x)

def r3_to_angles(x):
    return Hypersphere(dim=2).extrinsic_to_spherical(x)