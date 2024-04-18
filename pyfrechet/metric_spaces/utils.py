from joblib import Parallel, delayed
import numpy as np

def D_mat(M, y) -> np.ndarray:
    """
    Compute the matrix of squared-distances between the components of y (elements
    of a metric space M).
    
    Hence, in practice: 
    y is the .data attribute of a MetricData object.
    M is the .M attribute of a MetricData object (an object of class .MetricSpace)
    """
    N = y.shape[0] # Number of elements in y to compute pair-wise distances
    D = np.zeros((N,N)) # Initialize the matrix of distances
    # Since D is going to be symmetric, we compute the upper triangular part
    for i in range(N):
        for j in range(i+1,N):
            D[i, j] = M.d(M.index(y, i), M.index(y, j))**2
    return D + D.T

def D_mat_par(M, y, n_jobs: int=-2) -> np.ndarray:
    """
    Same purpose as D_mat but prepared for parallel computing with joblib module.
    Recommended for computationally expensive distance computations.

    n_jobs indicates the number of cores employed for parallel computing.
    """
    N = y.shape[0] # Number of elements in y to compute pair-wise distances

    # Function to be parallelised
    def calc(i, _y): 
        D = np.zeros(N) # Row vector of index i of the squared-distances matrix
        for j in range(i+1,N):
            D[j] = M.d(M.index(_y, i), M.index(_y, j))**2
        return D
    
    D = Parallel(n_jobs=n_jobs, verbose=0)(delayed(calc)(i, y) for i in range(N))
    D = np.r_[D] # Concatenate along the first axis the rows (to obtain final matrix)
    return D + D.T

def medoid_var(D: np.ndarray):
    """
    Compute the Frechet variance using the medoid estimation of the Frechet mean
    (with uniform weights). That is:

    medoid_var=\sum_{i=1}^{n} d( y_i, medoid )^2

    where medoid=\\argmin_{\omega\in\Omega_n} \sum_{i=1}^{n} d( y_i, medoid )^2.

    D will be the squared-distances matrix between elements of an array y
    (computed with either D_mat or D_mat_par)
    """
    return np.min(D.sum(axis=1))

def mat_sel_idx(D: np.ndarray, idx: list) -> np.ndarray:
    """
    Selects the submatrix of D containing the elements given by indices idx
    (select rows given by idx and columns given by idx too).

    Will be used to consider the distances of only the elements of a subsample 
    from the original data.
    """
    return D[idx,:][:,idx]

def mat_sel(D: np.ndarray, mask) -> np.ndarray:
    """
    Selects the submatrix of elements of D satisfying codition given by mask
    argument by means of mat_sel_idx() function.

    Will be employed to obtain the squared-distances matrix of subsamples of
    original data whose distances satisfy certain condition.
    """
    return mat_sel_idx(D, np.argwhere(mask)[:,0])

def coalesce_weights(w: np.ndarray, shaped) -> np.ndarray:
    """
    If weights are given, the funtion returns them. Otherwise,
    it create uniform weights (1/N,...,1/N).

    This function is going to be used to initialize weights when instantiate
    some classes where weights might not be given.
    shaped arguments must have a .shape attribute
    """
    N = shaped.shape[0]
    return w if not w is None else np.ones(N)/N


# def main():
#     D=np.array([[0,1],[2,3]])
#     print(mat_sel_idx(D, [0,1]))

# if __name__=='__main__':
#     main()

