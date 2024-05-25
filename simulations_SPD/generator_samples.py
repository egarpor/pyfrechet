import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
from typing import Union
import numpy as np
import pandas as pd
import pickle
from scipy.stats import wishart

def w_1(t):
    if t<=0.5:
        return np.cos(np.pi*t)**2
    else:
        return 0
    
def w_3(t):
    if t>=0.5:
        return np.cos(np.pi*(1-t))**2
    else:
        return 0
    
Sigma_1=np.array([[1, -0.7],
                  [-0.7, 0.5]])
Sigma_2=np.array([[1, 0],
                  [0, 1]])
Sigma_3=np.array([[0.5, 0.4],
                  [0.4, 1]])

def Sigma_true(t: float, 
           Sigmas: tuple):
    return w_1(t)*Sigmas[0]+(1-w_1(t)-w_3(t))*Sigmas[1]+w_3(t)*Sigmas[2]

def sim_regression_matrices(Sigmas: tuple,
                            size: int=1,
                            random_state: Union[None, int]=None,
                            df: int=2):
    np.random.seed(random_state)
    sample_t=np.sort(np.random.uniform(size=size))
    true_t=np.linspace(start=0, stop=1, num=size)
    sample_Y=[(1/df)*wishart(df=df, scale=Sigma_true(sample_t[k], Sigmas)).rvs(size=1) for k in range(sample_t.size)]
    true_Sigmat=[Sigma_true(true_t[k], Sigmas) for k in range(true_t.size)] 

    return {'sample': (sample_t, sample_Y),
            'true': (true_t, true_Sigmat)}

number_samples=100
test_size=100
sample_sizes=[50, 100, 200, 500]
sample_sizes=[size+test_size for size in sample_sizes]
nus=[1.5, 2, 2.5, 3, 4, 5, 6]

np.random.seed(1000)
for sample_size in sample_sizes:
    for nu in nus:
        for k in range(1, number_samples+1):
            sample=sim_regression_matrices((Sigma_1, Sigma_2, Sigma_3), 
                                           size=sample_size,  
                                           df=nu)
            
            filename=os.path.join(os.getcwd(), 'data/SPD_Samp'+str(k)+'_N'+ \
                                str(sample_size-test_size)+'_df'+ \
                                str(nus.index(nu)+1)+'.pkl')

            with open(filename, 'wb') as f:
                pickle.dump(sample, f)
            


