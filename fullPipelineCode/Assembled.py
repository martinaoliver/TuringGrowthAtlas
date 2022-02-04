from tqdm import tqdm
import sys

#########################################
################# SETUP #################
#########################################

# Function for parsing command line arguments

def parse_args(inputs):
    
    args = dict(
        num_nodes=2,
        num_diffusers=2,
        system_length=300,
        total_time=299,
        num_samples=100,
        growth=None
    )

    for a in inputs:

        if "-" in a:
            try:
                command_line_input = int(inputs[inputs.index(a) + 1])
            except:
                command_line_input = inputs[inputs.index(a) + 1]
            args[a[1:]] = command_line_input

    return args


#############################################
################# EXECUTION #################
#############################################

if __name__ == '__main__':

    args = parse_args(sys.argv)

    ################### PART ONE: ATLAS ########################
    
    from Part1_ATLAS import Atlas_Builder
    atlas = Atlas_Builder()
    atlas = atlas.create_adjacency_matrices(nodes = args['num_nodes'], diffusers = args['num_diffusers'])
    
    ################### PART TWO: PARAMETERS ###################
     
    from Part2_PARAMETERS import LHS  
    nsamples = args['num_samples']   
    sampler = LHS(nsamples = args['num_samples'])
    params = sampler.sample_parameters()
    
    ################### PART THREE: SOLVE ######################
    
    from Part3_SOLVE_v2 import Solver        
    for s in tqdm(range(nsamples)):   
        parameter_set = params.loc[s].to_dict() # Extract parameter set from dataframe.      
        for R in atlas:   
            conc = Solver.solve(p = parameter_set, topology = R, args = args)
            #Solver.plot_conc(conc)
        
        
        