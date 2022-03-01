# Author Xindong
# Date 2022/3/1 13:24

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


from solver_lsa import Solver


def run_solver(items):
    index, settings, args = items
    index_list = [i for i in index]

    try:
        LSA = Solver.solve(params=settings[0], topology=settings[1], **args)
        indexes = []
        for i in range(len(LSA)):
            new_index = index_list + [i]
            indexes.append(tuple(new_index))

        results = {i: {"LSA": l} for l in LSA}

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
        num_samples=100000,
        growth="None",
        growth_rate=0.1,
        dx=0.3,
        jobs=4
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

    ### LOAD PICKLE FILE
    print("Loading parameters")
    with open("", "rb") as f:
        params_and_arrays = pickle.load(f)

    items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]

    ################### PART THREE: SOLVE ######################
    print("Running solver...")

    results = multiprocess_wrapper(run_solver, items, 4)
    results = {k: v for d in results for k, v in d.items()}
    print("Saving results...")

    # # Saving results
    # with open("1t_results.pkl", "wb") as file:
    #     pickle.dump(results, file)
