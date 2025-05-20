import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

class Sphere(RiemannianManifold):
    def __init__(self, dim):
        super().__init__(Hypersphere(dim = dim))
        self.dim = dim
        self.extrinsic_dim = dim + 1

    def _d(self, x, y):
        """
        Compute the geodesic (great-circle) distance between two points on the sphere.
        The distance is scaled by the constant c_.
        """
        return self.manifold.metric.dist(x, y)

    #def _frechet_mean(self, y, w=None):
    #    extrinsic_mean = w.dot(y)
    #    return extrinsic_mean / np.linalg.norm(extrinsic_mean)

    def _frechet_mean(self, y, w):
        extrinsic_mean = w.dot(y)
        normalized_mean = extrinsic_mean / np.linalg.norm(extrinsic_mean)
        mean_calculator = FrechetMean(space=self.manifold)
        mean_calculator.set(
            init_point=normalized_mean,
            verbose=False
        )
        mean_calculator.fit(y, weights=w)
        return mean_calculator.estimate_

    def __str__(self):
        return f'Sphere(dim={self.manifold.dim})'

def r2_to_angle(x):
    return Hypersphere(dim=1).extrinsic_to_angle(x)

def r3_to_angles(x):
    return Hypersphere(dim=2).extrinsic_to_spherical(x)