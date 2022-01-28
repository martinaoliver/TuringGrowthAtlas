import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define dictionary containing parameters and their desired ranges.
# These default ranges come from Scholes 2019.

param_dict = {'prod_A' : [0.1, 100], \
              'prod_B' : [0.1, 100], \
              'deg_A' : [0.01,1], \
              'deg_B' : [0.01,1], \
              'max_A' : [0.1, 100], \
              'max_B' : [0.1, 100], \
              'K_AA' : [0.1, 100], \
              'K_AB' : [0.1, 100], \
              'K_BA' : [0.1, 100], \
              'K_BB' : [0.1, 100], \
              'D_A' : [1], \
              'D_B' : [0.001, 1000]}
    
# Function generates value from log distribution (0.001 to 1000).
    
def loguniform(low=-3, high=3, size=None): 
    return (10) ** (np.random.uniform(low, high, size))

# Function generates non-overlapping samples from a distribution (latin hypercube sampling). 
# (data = distribution, nsample = number of desired samples)
    
def lhs(data, nsample):
     P = 100*(np.random.permutation(nsample) + 1 - np.random.uniform(size=(nsample)))/nsample
     s = np.percentile(data, P)
     return s
 
# Function generates distribution range for each parameter and performs latin hypercube sampling.
 
def parameter_sampler(param_dict, nsamples = 200):

    param_sample_dict = {}
        
    for p in param_dict:
        
        if len(param_dict[p]) == 2: # If 2 numbers in parameter range we do sampling.
        
            distribution = []
            
            # Sample from log distribution until reaching 1000 points within the specified range.
            while len(distribution) < 1000:
                point = loguniform()
                if param_dict[p][0] <= point <= param_dict[p][1]:
                    distribution.append(point)
                    
            # Perform lhs using the resulting distribution.        
            samples = lhs(distribution, nsamples)
            
            param_sample_dict[p] = samples
            
        elif len(param_dict[p]) == 1: # If one number in parameter range it is constant.
            
            param_sample_dict[p] = param_dict[p][0]
        
    df = pd.DataFrame(param_sample_dict)
    
    return df

print(parameter_sampler(param_dict))
