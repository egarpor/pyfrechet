import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.euclidean import Euclidean

class two_euclidean(RiemannianManifold):
    def __init__(self, dim):
        super().__init__(Euclidean(dim=dim))
        self.dim = dim
        self.extrinsic_dim = dim

    def __str__(self):
        return f'Euclidean(dim={self.manifold.dim})'