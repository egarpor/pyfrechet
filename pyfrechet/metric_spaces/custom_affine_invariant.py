import numpy as np
from pyfrechet.metric_spaces import MetricSpace 
from scipy.linalg import eigvals
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean

class CustomAffineInvariant(MetricSpace):
    """
    Affine-invariant Riemannian metric space for dxd SPD matrices.
    
    The distance between two SPD matrices A and B is defined as:
    d(A, B) = || log(A^(-1/2) B A^(-1/2)) ||_F

    The Fréchet mean of a set of SPD matrices {X_i} is estimated from the sample using the function mean_riemann 
    from pyriemann.utils.mean, based on the papers:
    .. [1] `Principal geodesic analysis for the study of nonlinear statistics
        of shape
        <https://ieeexplore.ieee.org/document/1318725>`_
        P.T. Fletcher, C. Lu, S. M. Pizer, S. Joshi.
        IEEE Trans Med Imaging, 2004, 23(8), pp. 995-1005
    .. [2] `A differential geometric approach to the geometric mean of
        symmetric positive-definite matrices
        <https://epubs.siam.org/doi/10.1137/S0895479803436937>`_
        M. Moakher. SIAM J Matrix Anal Appl, 2005, 26 (3), pp. 735-747
    """

    def __init__(self, dim):
        self.dim = dim 
        self.manifold = SPDMatrices(n=dim)

    def _d(self, S1, S2):
        """
        Computes the affine-invariant Riemannian metric (AIRM) between two SPD matrices S1 and S2.

        Parameters:
            S1 (ndarray): Symmetric positive definite matrix of shape dxd.
            S2 (ndarray): Symmetric positive definite matrix of shape dxd.

        Returns:
            float: The affine-invariant Riemannian metric distance between S1 and S2.
        """
        # Note: manifold.dist(S1, S2) returns the same output, but is slower. Although we still use
        # geomstats for the Fréchet mean, the medoids only require computing the distances, so using this function saves time.
        # The Fréchet mean could also be computed using mean_riemann() from pyriemann.utils.mean. 

        # Compute the matrix S1^{-1}S2
        inv_S1_S2 = np.linalg.solve(S1, S2)

        # Compute the eigenvalues of S1^{-1}S2
        eigenvalues = eigvals(inv_S1_S2)

        # Compute the log of eigenvalues and sum of their squares
        log_eigenvalues = np.log(eigenvalues.real)  # Ensure real part is taken

        return np.sqrt(np.sum(log_eigenvalues**2))

    def _frechet_mean(self, y, w):
        mean = FrechetMean(metric=self.manifold.metric)
        mean.fit(y, weights=w)
        return mean.estimate_

    def __str__(self):
        return f'CustomAffineInvariant({self.dim}x{self.dim})'
    

import numpy as np

class SPDVectorizer:
    """Wraps SPD matrices for sklearn compatibility by vectorizing them."""
    def __init__(self, metric_space):
        self.metric_space = metric_space  # Instance of CustomAffineInvariant

    def vectorize(self, matrix):
        """Converts a 2x2 SPD matrix to a 3D vector."""
        return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 1]])

    def devectorize(self, vector):
        """Converts a 3D vector back to a 2x2 SPD matrix."""
        return np.array([[vector[0], vector[2]], 
                         [vector[2], vector[1]]])

    def dist(self, v1, v2):
        """Computes distance using SPD matrices."""
        return self.metric_space.d(self.devectorize(v1), self.devectorize(v2))

    def frechet_mean(self, vectors, weights):
        """Computes the Fréchet mean using SPD matrices."""
        matrices = np.array([self.devectorize(v) for v in vectors])
        return self.vectorize(self.metric_space._frechet_mean(matrices, weights))
