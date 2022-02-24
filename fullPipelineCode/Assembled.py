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
        p.close()
        p.join()
    return results


from solver import Solver


def run_solver(items):
    index, settings, args = items
    index_list = [i for i in index]

    try:
        concs, steadystates, LSA = Solver.solve(params=settings[0], topology=settings[1], **args)
        indexes = []
        for i in range(len(concs)):
            new_index = index_list + [i]
            indexes.append(tuple(new_index))

        # return concentration
        concs = list(zip(indexes, concs))
        # return steady states
        steadystates = list(zip(indexes, steadystates))
        # return LSA
        LSA = list(zip(indexes, LSA))

        return concs

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
    atlas = atlas.create_adjacency_matrices(nodes=args['num_nodes'], diffusers=args['num_diffusers'])
    atlas = {0:np.array([[1,1],[-1,0]])}

    ################### PART TWO: PARAMETERS ###################
    print("Sampling parameters...")
    from parameters import LHS

    nsamples = args['num_samples']
    sampler = LHS(nsamples=args['num_samples'])
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

    ################### PART THREE: SOLVE ######################

    print("Running solver...")

    results = multiprocess_wrapper(run_solver, items, 4)

    print("Saving results...")

    # Saving results
    with open("/1t_results.pkl", "wb") as file:
        pickle.dump(results, file)
