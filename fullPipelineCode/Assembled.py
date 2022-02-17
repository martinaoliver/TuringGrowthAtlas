from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain
import pickle



#########################################
################# SETUP #################
#########################################

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
        x = p.imap(function, items)
        p.close()
        p.join()
    return results

from solver import Solver
def run_solver(items):
    index, settings, args = items
    index_list = [i for i in index]

    try:
        concs, steadystates = Solver.solve(params = settings[0], topology = settings[1],  **args)
        indexes = []
        for i in range(len(concs)):
            new_index = index_list + [i]
            indexes.append(tuple(new_index))
        concs = list(zip(indexes, concs))

        steadystates = list(zip(indexes, steadystates))
        return concs

    except:
        traceback.print_exc()
        raise



# Function for parsing command line arguments

def parse_args(inputs):

    args = dict(
        num_nodes=2,
        num_diffusers=2,
        system_length=200,
        total_time=199,
        num_samples=200,
        growth="linear"
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

    print("Parsing inputs...")
    args = parse_args(sys.argv)

    ################### PART ONE: ATLAS ########################
    print("Building atlas...")
    from atlas import Atlas
    atlas = Atlas()
    atlas = atlas.create_adjacency_matrices(nodes = args['num_nodes'], diffusers = args['num_diffusers'])

    ################### PART TWO: PARAMETERS ###################
    print("Sampling parameters...")
    from parameters import LHS
    nsamples = args['num_samples']
    sampler = LHS(nsamples = args['num_samples'])
    params = sampler.sample_parameters()

    # Prepare grid space, rate, and time
    args["J"] = args["system_length"]
    args["dx"] = args["J"] / (args["J"] - 1.)
    args["num_timepoints"] = int(10. * args["total_time"])
    args["dt"] = args["total_time"] / (args["num_timepoints"] - 1.)

    for p in params:
        # Calculate alpha values for each species.
        for diff in ['diffusion_x', 'diffusion_y']:
            params[p][f"alphan_{diff[-1]}"] = Solver.calculate_alpha(params[p][diff], **args)


    # Join altas and params
    indexes = product(params.keys(), atlas.keys())
    combinations = product(params.values(), atlas.values())

    params_and_arrays = {index: combination for index,combination in zip(indexes, combinations)}

    items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]
    items = items[:2]

    ################### PART THREE: SOLVE ######################
    print("Running solver...")
    results = multiprocess_wrapper(run_solver, items, 4)

    with open("/Users/sammoney-kyrle/Google Drive/TuringGrowthAtlas/results/2d_100_params_results.pkl", "wb") as file:
        pickle.dump(results, file)
