import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
from scipy.special import digamma
from scipy.stats import wishart
import pickle

def Sigma_t(t_array, Sigma_array):
    """Provides an array with the matrices given by a regression model that interpolates between four matrices."""  
    """The regression starts with Sigma_1 and then goes to Sigma_2 and Sigma_3 and ends in Sigma_4."""
    
    # Define time intervals for interpolation
    t_array = np.array(t_array)
    t_array = t_array[:, None, None]

    # Return the interpolated matrices
    return np.where(t_array < 0.5, np.cos(np.pi*t_array)**2 * Sigma_array[0] + (1 - np.cos(np.pi*(1-t_array))**2) * Sigma_array[1], 0) + np.where(t_array >= 0.5, (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1] + np.cos(np.pi*(1-t_array))**2 * Sigma_array[2], 0)



def sim_regression_matrices(Sigmas: tuple,
                            t: np.array,
                            df: int=2):
    t = np.array(t)
    
    q = Sigmas[0].shape[0]

    c_dq = 2 * np.exp((1 / q) * sum( digamma((df - np.arange(1, q + 1) + 1 ) / 2) ))
    sigma_t = Sigma_t(t, Sigmas)
    sample_Y = [wishart( df=df, scale = sigma_t[k]/c_dq ).rvs( size=1 ) for k in range(t.shape[0])]
    return {'t': t, 'y': sample_Y} 
    

# Define the matrices to interpolate 
Sigma_1 = np.array([[1, -0.6],
                  [-0.6, 0.5]])
Sigma_2 = np.array([[1, 0],
                  [0, 1]])
Sigma_3 = np.array([[0.5, 0.4],
                  [0.4, 1]])

n_samples = 100
sample_sizes = [50, 100, 200, 500]
sample_sizes = [size for size in sample_sizes]
dfs = [5, 10, 15]

# For each combination of sample size and degrees of freedom, generate n_samples samples

np.random.seed(1000)
for sample_size in sample_sizes:
    for df in dfs:
        for k in range(1, n_samples+1):
            sample_t = np.random.uniform(size=sample_size)
            sample = sim_regression_matrices(Sigmas = (Sigma_1, Sigma_2, Sigma_3), 
                                           t = sample_t,  
                                           df = df)
            
            filename = os.path.join(os.getcwd(), 'simulations_SPD', 'data', 'SPD_Samp' + str(k)+'_N'+ \
                                str(sample_size)+ '_df' + \
                                str(df)+'.pkl')

            with open(filename, 'wb') as f:
                pickle.dump(sample, f)
            


