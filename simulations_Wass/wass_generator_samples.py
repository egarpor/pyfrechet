import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
sys.path.append(os.getcwd())
from typing import Union
import numpy as np
import pickle
import json
from pyfrechet.metric_spaces import wasserstein_1d as ws
from scipy import stats 

GRID = np.linspace(0.01, 0.99, 100)
STD_NORMAL_Q = stats.norm.ppf(GRID)


def sample_linear_transport(x, sig=1, gam=0.5):
    gam = np.random.gamma(0.5, 0.5)
    sig = np.random.exponential(0.5)
    Q0 = gam - np.log(1 + x) + (sig + x**2) * STD_NORMAL_Q
    return Q0 
    
def gen_data(N):
    # We know the values of Q in a grid, and we interpolate to estimate the values of Q in the new grid
    x = np.random.uniform(0,1, N)
    y = np.array([ sample_linear_transport(x[i]) for i in range(N)])
    
    return {'x': x, 'y': y}


n_samples = 500
sample_size = 400

np.random.seed(1000)

for k in range(1, n_samples+1):
    sample = gen_data(sample_size)

    filename = os.path.join(os.getcwd(), 'simulations_Wass', 'wass_data', 'WASS_Samp' + str(k) + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(sample, f)