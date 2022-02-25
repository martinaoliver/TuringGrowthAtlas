from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain
import pickle
import numpy as np


#########################################
################# SETUP #################
#########################################

# Import results.
infile = open('1t_results.pkl','rb')
results_dict = pickle.load(infile)
infile.close()

# Import parameters.
infile = open('parameters.pkl','rb')
parameter_data = pickle.load(infile)
infile.close()

# Loop through results to identify hits.
hits = {}
for i in results_dict:
    if results_dict[i]['Fourier'] == False:
        hits[i] = results_dict[i]
        
        

def multiprocess_wrapper(function, items, cpu):
    ###########################################
    # Function: Parallelise input function    #
    #                                         #
    # Inputs: Function to parallelise         #
    # (def score),                            #
    # list of tuples as function input,       #
    # number of threads to parallelise across #
    #                                         #
    # Output: List of returned results        #
    ###########################################

    processes = min(cpu, mp.cpu_count())

    with mp.Pool(processes) as p:
        results = list(tqdm(p.imap(function, items), total=len(items)))
        p.close()
        p.join()
    return results


from nb_solver_lsa import Solver


def run_solver(items):
    index, settings, args = items
    index_list = [i for i in index]

    try:
        concs, steadystates, LSA, fourier = Solver.solve(params=settings[0], topology=settings[1], **args)
        indexes = []
        for i in range(len(concs)):
            new_index = index_list + [i]
            indexes.append(tuple(new_index))


        results = {i: {"concs": c, "steadystate":s, "LSA":l, "Fourier":f} for i,c,s,l,f in zip(indexes, concs, steadystates, LSA, fourier)}

        return results

    except:
        traceback.print_exc()
        raise


# Function for parsing command line arguments

def parse_args(inputs):
    args = dict(
        num_nodes=2,
        num_diffusers=2,
        system_length=50,
        total_time=1000,
        num_samples=100,
        growth="None",
        growth_rate=0.1,
        dx=0.3,
        jobs=4,
        results='1t_results.pkl',
        parameters='parameters.pkl'
    )

    for a in inputs:

        if "-" in a:
            try:
                command_line_input = int(inputs[inputs.index(a) + 1])
            except:
                command_line_input = inputs[inputs.index(a) + 1]
            args[a[1:]] = command_line_input


    if args["growth"] == "None":
        args["growth"] = None

    return args


#############################################
################# EXECUTION #################
#############################################

if __name__ == '__main__':

    print("Parsing inputs...")
    args = parse_args(sys.argv)
    
    # Run the solver for each hit and save in a separate file.
    
    # Import results.
    infile = open(args['results'],'rb')
    results_dict = pickle.load(infile)
    infile.close()
    
    # Import parameters.
    infile = open(args['parameters'],'rb')
    parameter_data = pickle.load(infile)
    infile.close()
    
    # Loop through results to identify hits.
    hits = {}
    for i in results_dict:
        if results_dict[i]['Fourier'] == False:
            hits[i] = results_dict[i]
    
    for hit in hits:

        ################### PART ONE: ATLAS ########################
        print("Extracting topology...")
        atlas = {0:(parameter_data[(hit[0],0)][1])}
    
        ################### PART TWO: PARAMETERS ###################
        print("Sampling parameters...")
        from nb_parameters import LHS
    
        nsamples = args['num_samples']
        
        hit_params = parameter_data[(hit[0],0)][0]
        hit_params.pop('alphan_x')
        hit_params.pop('alphan_y')
        
        sampler = LHS(nsamples=args['num_samples'], params = hit_params)
        params = sampler.sample_parameters()
    
        # Prepare grid space, rate, and time
        args["J"] = args["system_length"]
        args["num_timepoints"] = int(12. * args["total_time"])
        args["dt"] = args["total_time"] / (args["num_timepoints"]-1)
    
        # Calculate alpha values for each species.
        for p in params:
            params[p]['alphan_x'] = Solver.calculate_alpha(params[p]['diffusion_x'], **args)
            params[p]['alphan_y'] = Solver.calculate_alpha(params[p]['diffusion_y'], **args)
    
        # Join altas and params
        indexes = product(params.keys(), atlas.keys())
        combinations = product(params.values(), atlas.values())
    
        params_and_arrays = {index: combination for index, combination in zip(indexes, combinations)}
    
        items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]
    
        print("Saving parameters...")
        with open(f"Parameters/{hit}_neighbourhood_parameters.pkl", "wb") as file:
            pickle.dump(params_and_arrays, file)
    
        ################### PART THREE: SOLVE ######################
        print("Running solver...")
    
        results = multiprocess_wrapper(run_solver, items, 4)
        results = {k: v for d in results for k, v in d.items()}
        print("Saving results...")
    
        # Saving results
        with open(f"Results/{hit}_results.pkl", "wb") as file:
            pickle.dump(results, file)
