from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain
import pickle
from pprint import pprint
import numpy as np



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
        system_length=100,
        total_time=100,
        num_samples=100000,
        growth="None",
        growth_rate=0.1,
        initial_dx=0.1,
        jobs=4
    )

    for a in inputs:

        if "-" in a:
            try:
                command_line_input = float(inputs[inputs.index(a) + 1])
            except:
                command_line_input = inputs[inputs.index(a) + 1]
            args[a[1:]] = command_line_input
    # Prepare grid space, rate, and time
    args["J"] = args["system_length"]
    args["num_timepoints"] = int(10. * args["total_time"])
    args["dt"] = args["total_time"] / (args["num_timepoints"] - 1.)
    hours = [i/(args["num_timepoints"] / args["total_time"]) for i in range(args["num_timepoints"])]

    if args["growth"] == "None":
        args["growth"] = None
        args["dx"] = [args["initial_dx"]]

    if args["growth"] == "exponential":
        args["dx"] = [Solver.exponential_growth(args["initial_dx"], dt=args["dt"], t=i, s=args["growth_rate"]) for i in hours]

    if args["growth"] == "linear":
        args["dx"] = [Solver.linear_growth(args["initial_dx"], dt=args["dt"], t=i, s=args["growth_rate"]) for i in hours]

    if args["growth"] == "weird":
        args["dx"] = [Solver.weird_growth(args["initial_dx"], dt=args["dt"], t=i, s=args["growth_rate"]) for i in hours]


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
    atlas = {0:np.array([[1,1],[-1,0]])}

    ################### PART TWO: PARAMETERS ###################
    print("Sampling parameters...")
    from parameters import LHS
    nsamples = args['num_samples']
    sampler = LHS(nsamples = args['num_samples'])
    params = sampler.sample_parameters()


    for p in params:
        # Calculate alpha values for each species.
        for diff in ['diffusion_x', 'diffusion_y']:
            params[p][f"alphan_{diff[-1]}"] = [Solver.calculate_alpha(params[p][diff], args["dt"], i) for i in args["dx"]]



    # Join altas and params
    indexes = product(params.keys(), atlas.keys())
    combinations = product(params.values(), atlas.values())

    params_and_arrays = {index: combination for index,combination in zip(indexes, combinations)}

    items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]

    ################### PART THREE: SOLVE ######################
    print("Running solver...")

    results = multiprocess_wrapper(run_solver, items, args["jobs"])

    print("Saving results...")

    with open("/Users/sammoney-kyrle/Google Drive/TuringGrowthAtlas/results/2d_100_params_results.pkl", "wb") as file:
        pickle.dump(results, file)
