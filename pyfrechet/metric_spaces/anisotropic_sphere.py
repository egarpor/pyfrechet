import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from .metric_space import MetricSpace


class AnisotropicSphere(MetricSpace):
    def __init__(self, dim, c_lambda_):
        #super().__init__(Hypersphere(dim = dim))
        self.dim = dim
        self.extrinsic_dim = dim + 1
        self.manifold = Hypersphere(dim=dim)
        # Riemannian metric is overridden (no structure of Riemannian manifold)  
        self.dist_override = True
        #c_lambda is a tuple of two elements: [c, lambda_]
        # c is the constant that normalizes the distance, lambda_ is the penalty in z-axis
        # assert that lambda_ > -1
        if c_lambda_[1] <= -1:
            raise ValueError("lambda_ must be greater than -1")
        self.weights = np.hstack([np.repeat(1, self.extrinsic_dim - 1), 1 + c_lambda_[1]])
        self.c_ = c_lambda_[0]

    def _d(self, x, y):
        """
        Computes anisotropic distances row-wise between X and Y (both n x 3 arrays).
        
        Parameters:
        - X, Y: numpy arrays of shape (n, 3), each row is a point on the unit sphere
        - lambda_: anisotropy coefficient; more penalty in z-direction

        Returns:
        - distances: numpy array of shape (n,), the anisotropic distances
        """
        diff = x - y  # shape (n, 3)
        weighted_diff = diff**2 * self.weights
        return self.c_*np.sqrt(np.sum(weighted_diff.reshape(-1, weighted_diff.shape[-1]), axis=1))

    def _frechet_mean(self, y, w=None):
        # Important: it is better to use medoids here!
        extrinsic_mean = w.dot(y)
        return extrinsic_mean / np.linalg.norm(extrinsic_mean)

    def __str__(self):
        return f'Sphere, anisotropic distance (dim={self.manifold.dim}, lambda_={self.weights[-1] - 1}, c_={self.c_})'
 
def r2_to_angle(x):
    return Hypersphere(dim=1).extrinsic_to_angle(x)

def r3_to_angles(x):
    return Hypersphere(dim=2).extrinsic_to_spherical(x)