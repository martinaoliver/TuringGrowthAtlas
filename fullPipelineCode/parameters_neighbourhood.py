import numpy as np
import pandas as pd

class LHS: # Defines parameter sampling

    def __init__(self, nsamples = 200, params = None):
        self.nsamples = nsamples
        self.params = {
            'production_x'  :  5,    # Production rate.
            'production_y'  :  5,
            'degradation_x' :  5,    # Degradation rate.
            'degradation_y' :  5,
            'max_conc_x'    :  5,    # Saturation.
            'max_conc_y'    :  5,
            'k_xx'          :  5,    # Reaction rates.
            'k_xy'          :  5,
            'k_yy'          :  5,
            'k_yx'          :  5,
            'diffusion_x'   :  5,    # Diffusion constants.
            'diffusion_y'   :  5,
            }

    # Function generates non-overlapping samples from a distribution (latin hypercube sampling).
    # data input is a distribution, which in this case will be a gaussian centered around the parameter value.

    def lhs(self,data):
         P = 100*(np.random.permutation(self.nsamples) + 1 - np.random.uniform(size=(self.nsamples)))/self.nsamples
         s = np.percentile(data, P)
         return s

    # Function generates distribution range for each parameter and performs latin hypercube sampling.
    def sample_parameters(self):

        param_sample_dict = {}

        for p in self.params:
            
            # Make gaussian distribution using the parameter value.
            distribution = np.random.normal(self.params[p], self.params[p]/4)

            # Perform lhs using the resulting distribution.
            samples = self.lhs(distribution)

            param_sample_dict[p] = samples.tolist()

        # Convert dictionary of lists into dictionary of dictionaries
        samples = {count: dict(zip(param_sample_dict, i)) for count, i in enumerate(zip(*param_sample_dict.values()))}

        return samples