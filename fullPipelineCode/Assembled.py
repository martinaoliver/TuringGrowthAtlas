from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain
import pickle
import numpy as np
import datetime


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
        total_time=1500,
        num_samples=100000,
        growth=None,
        growth_rate=0.1,
        dx=0.6,
        jobs=4,
        neighbourhood=False,
        upload_params=False,
        results_file=None,
        param_file=None,
        save_all=False
    )

    for a in inputs:

        if "-" in a:
            try:
                command_line_input = int(inputs[inputs.index(a) + 1])
            except:
                command_line_input = inputs[inputs.index(a) + 1]
            args[a[1:]] = command_line_input

    if args["save_all"]=="True":
        args["save_all"]=True

    return args


#############################################
################# EXECUTION #################
#############################################

if __name__ == '__main__':

    print("Parsing inputs...")

    args = parse_args(sys.argv)

    # Not doing neighbourhood searching.
    if args['neighbourhood'] == False:

        ################### PART ONE: ATLAS ########################
        print("Building atlas...")
        # from atlas import Atlas
        #
        # atlas = Atlas()
        # atlas = atlas.create_adjacency_matrices(nodes=args['num_nodes'], diffusers=args['num_diffusers'])
        atlas = {0:np.array([[1,1],[-1,0]])}

        ################### PART TWO: PARAMETERS ###################

        # Prepare grid space, rate, and time
        args["J"] = args["system_length"]
        args["num_timepoints"] = int(12. * args["total_time"])
        args["dt"] = args["total_time"] / (args["num_timepoints"]-1)

        # If a pre-existing parameter file is to be used then load it.
        if args['upload_params'] == True:

            infile = open(args['upload_params'], 'rb')
            parameter_data = pickle.load(infile)
            infile.close()
            params_and_arrays = parameter_data

        # Otherwise generate parameters from scratch.
        else:
            print("Sampling parameters...")

            from parameters import LHS
            nsamples = args['num_samples']
            sampler = LHS(nsamples=args['num_samples'])
            params = sampler.sample_parameters()

            # Calculate alpha values for each species.
            for p in params:
                params[p]['alphan_x'] = Solver.calculate_alpha(params[p]['diffusion_x'], **args)
                params[p]['alphan_y'] = Solver.calculate_alpha(params[p]['diffusion_y'], **args)

            # Join altas and params
            indexes = product(params.keys(), atlas.keys())
            combinations = product(params.values(), atlas.values())

            params_and_arrays = {index: combination for index, combination in zip(indexes, combinations)}

            print("Saving parameters...")
            with open("parameters.pkl", "wb") as file:
                pickle.dump(params_and_arrays, file)

        items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]

        ################### PART THREE: SOLVE ######################
        timestamp = str(datetime.datetime.now())
        timestamp = timestamp.replace(':', '-')[:16]
        timestamp = timestamp.replace(' ', '_')

        chunks = [items[x:x+int(len(items)/10)] for x in range(0, len(items), int(len(items)/10))]
        results_dict = dict()
        for i in chunks:
            print("Running solver...")
            results = multiprocess_wrapper(run_solver, i, args["jobs"])
            for d in results:
                for k, v in d.items():
                    results_dict[k] = v
            print("Saving results...")

            # Saving results
            with open(f"{args['growth']}_results.pkl", "wb") as file:
                pickle.dump(results_dict, file)


    # If you wish to do neighbourhood searching of previous results.
    if args['neighbourhood']:

        # Import prior results.
        infile = open(args['results_file'], 'rb')
        results_dict = pickle.load(infile)
        infile.close()

        # Import parameters.
        infile = open(args['param_file'], 'rb')
        parameter_data = pickle.load(infile)
        infile.close()

        # Loop through results to identify hits.
        hits = {}
        for i in results_dict:
            if results_dict[i]['Fourier'][0] and results_dict[i]['Fourier'][1]:
                hits[i] = results_dict[i]

        for hit in hits:

            ################### PART ONE: ATLAS ########################
            print("Extracting topology...")
            atlas = {0:(parameter_data[(hit[0],0)][1])}

            ################### PART TWO: PARAMETERS ###################
            print("Sampling parameters...")
            from parameters import LHS_neighbourhood

            nsamples = args['num_samples']

            hit_params = parameter_data[(hit[0],0)][0]
            hit_params.pop('alphan_x')
            hit_params.pop('alphan_y')

            sampler = LHS_neighbourhood(nsamples=args['num_samples'], params = hit_params)
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
            with open(f"{hit}_neighbourhood_parameters.pkl", "wb") as file:
                pickle.dump(params_and_arrays, file)

            ################### PART THREE: SOLVE ######################

            timestamp = str(datetime.datetime.now())
            timestamp = timestamp.replace(':', '-')[:16]
            timestamp = timestamp.replace(' ', '_')

            chunks = [items[x:x+int(len(items)/10)] for x in range(0, len(items), int(len(items)/10))]
            results_dict = dict()
            for i in chunks:
                print("Running solver...")
                results = multiprocess_wrapper(run_solver, i, args["jobs"])
                for d in results:
                    for k, v in d.items():
                        results_dict[k] = v
                print("Saving results...")

                # Saving results
                with open(f"{hit}_results.pkl", "wb") as file:
                    pickle.dump(results_dict, file)
