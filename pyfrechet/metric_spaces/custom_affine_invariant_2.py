import numpy as np
from scipy.linalg import eigvals
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from pyfrechet.metric_spaces import MetricSpace

class CustomAffineInvariant_2(MetricSpace):
    """
    A vectorized version of CustomAffineInvariant for use with scikit-learn.

    It represents 2x2 SPD matrices as 3D vectors:
        [a, b, c]  <-->  [[a, c], 
                           [c, b]]
    
    Internally, computations are performed using SPD matrices, 
    but externally, scikit-learn sees only vectors.
    """

    def __init__(self, dim=2):
        if dim != 2:
            raise ValueError("CustomAffineInvariant_2 only supports 2x2 SPD matrices (dim=2).")
        self.dim = dim
        self.manifold = SPDMatrices(n=dim)

    def vectorize(self, matrix):
        """Converts a 2x2 SPD matrix to a 3D vector."""
        return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 1]])

    def devectorize(self, vector):
        """Converts a 3D vector back to a 2x2 SPD matrix."""
        return np.array([[vector[0], vector[2]], 
                         [vector[2], vector[1]]])

    def _d(self, v1, v2):
        """
        Computes the affine-invariant Riemannian distance between two SPD matrices.

        Parameters:
            v1, v2: 1D numpy arrays of shape (3,).
        Returns:
            float: The affine-invariant distance.
        """
        S1 = self.devectorize(v1)
        S2 = self.devectorize(v2)

        # Compute S1^(-1) S2
        inv_S1_S2 = np.linalg.solve(S1, S2)

        # Compute the eigenvalues of S1^(-1) S2
        eigenvalues = eigvals(inv_S1_S2)

        # Compute the log of eigenvalues and sum of their squares
        log_eigenvalues = np.log(eigenvalues.real)  # Ensure real part is taken

        return np.sqrt(np.sum(log_eigenvalues**2))

    def _frechet_mean(self, y, w):
        """
        Computes the Fréchet mean of SPD matrices using vectorized inputs.

        Parameters:
            y (array-like): A list/array of shape (n_samples, 3), where each row is a vectorized SPD matrix.
            w (array-like): Weights associated with each sample.

        Returns:
            numpy array of shape (3,): The vectorized Fréchet mean.
        """
        matrices = np.array([self.devectorize(v) for v in y])  # Convert to matrices
        mean = FrechetMean(metric=self.manifold.metric, point_type='matrix', verbose=False)
        mean.fit(matrices, weights=w)
        return self.vectorize(mean.estimate_)  # Convert back to vector

    def __str__(self):
        return f'CustomAffineInvariant_2({self.dim}x{self.dim}, vectorized)'
