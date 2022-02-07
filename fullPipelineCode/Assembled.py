from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product


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
        results = dict(p.imap(function, items))
        x = p.imap(function, items)
        p.close()
        p.join()
    return results

from Part3_SOLVE_v2 import Solver
def run_solver(items):
    items, args = items
    try:
        concs_list = []
        for count, item in enumerate(items):
            conc, ss = Solver.solve(p = item[0], topology = item[1], args = args)



    except:
        traceback.print_exc()
        raise



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

    print("Parsing inputs...")
    args = parse_args(sys.argv)

    ################### PART ONE: ATLAS ########################
    print("Building atlas...")
    from Part1_ATLAS import Atlas_Builder
    atlas = Atlas_Builder()
    atlas = atlas.create_adjacency_matrices(nodes = args['num_nodes'], diffusers = args['num_diffusers'])

    ################### PART TWO: PARAMETERS ###################
    print("Sampling parameters...")
    from Part2_PARAMETERS import LHS
    nsamples = args['num_samples']
    sampler = LHS(nsamples = args['num_samples'])
    params = sampler.sample_parameters()

    # Join altas and params
    indexes = product(params.keys(), atlas.keys())
    combinations = product(params.values(), atlas.values())

    params_and_arrays = {index: combination for index,combination in zip(indexes, combinations)}

    items = [(pa, args) for pa in params_and_arrays]

    ################### PART THREE: SOLVE ######################
    print("Running solver...")
    results = multiprocess_wrapper(run_solver, items, 2)
