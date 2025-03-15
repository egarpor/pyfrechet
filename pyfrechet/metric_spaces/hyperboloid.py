import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.hyperboloid import Hyperboloid

class H2(RiemannianManifold):
    def __init__(self, dim):
        super().__init__(Hyperboloid(dim = dim))

    # def _frechet_mean(self, y, w=None):
    #     extrinsic_mean = w.dot(y)
    #     return extrinsic_mean / np.linalg.norm(extrinsic_mean)

    def __str__(self):
        return f'Hyperboloid(dim = {self.manifold.dim})'
 
def extrinsic_to_intrinsic(x):
    return Hyperboloid(dim = dim).extrinsic_to_intrinsic_coords(x)