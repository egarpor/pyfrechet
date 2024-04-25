import numpy as np
from geomstats.learning.frechet_mean import FrechetMean
# from geomstats.geometry.product_manifold import ProductManifold, NFoldManifold, NFoldMetric
from geomstats.geometry.hypersphere import Hypersphere
from .metric_space import MetricSpace
from .sphere import Sphere

class Torus(MetricSpace):
    def __init__(self, dim):
        self.dim=dim
        # self.manifold=ProductManifold(manifolds=[Hypersphere(dim=1, default_coords_type='intrinsic') for _ in range(self.dim)],
        #                             default_point_type='vector')
        # self.manifold=Landmarks(ambient_manifold=Hypersphere(dim=1, default_coords_type='intrinsic'),
        #                         k_landmarks=self.dim)
        self.marginal_manifold=Sphere(dim=1)
    
    def _d(self, x, y):
        aux=Hypersphere(dim=1, default_coords_type='intrinsic')
        
        if x.ndim==1:
            marginal_distances=[]
            for j in range(self.dim):
                x_marginal=aux.angle_to_extrinsic(x[j])
                y_marginal=aux.angle_to_extrinsic(y[j])
                marginal_distances.append(self.marginal_manifold.d(x_marginal, y_marginal))
            return np.sqrt(np.sum(np.array(marginal_distances)**2))
        else:
            marginal_distances=np.zeros(shape=(x.shape[0], self.dim))
            for j in range(self.dim):
                x_marginal=aux.angle_to_extrinsic(x[:,j])
                y_marginal=aux.angle_to_extrinsic(y[:,j])
                marginal_distances[:,j]=self.marginal_manifold.d(x_marginal, y_marginal)
            return np.sqrt(np.sum(np.array(marginal_distances)**2, axis=1))

    def _frechet_mean(self, y, w=None):
        mean=[]
        aux=Hypersphere(dim=1, default_coords_type='intrinsic')
        for j in range(self.dim):
            y_marginal=aux.angle_to_extrinsic(y[:,j].flatten())
            mean_extrinsic=aux.extrinsic_to_angle(self.marginal_manifold._frechet_mean(y_marginal, w=w))
            mean.append(mean_extrinsic)
        return np.array(mean)
    
    def __str__(self) -> str:
        return f'Torus(dim={self.dim})'
 


