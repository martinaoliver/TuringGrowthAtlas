from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain



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
        results = p.imap(function, items)
        x = p.imap(function, items)
        p.close()
        p.join()
    return results

from solver import Solver
def run_solver(items):
    index, params, args = items
    index_list = [i for i in index]

    try:
        concs, ss = Solver.solve(p = params[0], topology = params[1], args = args)
        indexes = []
        for i in range(len(concs)):
            new_index = index_list + [i]
            indexes.append(tuple(new_index))
        concs = list(zip(indexes, concs))

        #ss = list(zip(indexes, ss))
        return concs #ss

    except:
        traceback.print_exc()
        raise



# Function for parsing command line arguments

def parse_args(inputs):

    args = dict(
        num_nodes=2,
        num_diffusers=2,
        system_length=100,
        total_time=99,
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
    from atlas import Atlas
    atlas = Atlas_Builder()
    atlas = atlas.create_adjacency_matrices(nodes = args['num_nodes'], diffusers = args['num_diffusers'])

    ################### PART TWO: PARAMETERS ###################
    print("Sampling parameters...")
    from parameters import LHS
    nsamples = args['num_samples']
    sampler = LHS(nsamples = args['num_samples'])
    params = sampler.sample_parameters()

    # Join altas and params
    indexes = product(params.keys(), atlas.keys())
    combinations = product(params.values(), atlas.values())

    params_and_arrays = {index: combination for index,combination in zip(indexes, combinations)}

    items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]


    ################### PART THREE: SOLVE ######################
    print("Running solver...")
    results = multiprocess_wrapper(run_solver, items, 1)
    results = {key:value for key, value in chain(*results)}
    print(results)
