from tqdm import tqdm
import sys
import multiprocessing as mp
import traceback
from itertools import product, chain
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        system_length=100,
        total_time=1500,
        num_samples=100000,
        growth=None,
        growth_rate=0.1,
        dx=0.2,
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
        atlas = {0:np.array([[1,-1],[1,0]])}

        ################### PART TWO: PARAMETERS ###################

        # Prepare grid space, rate, and time
        args["J"] = args["system_length"]
        args["num_timepoints"] = int(12. * args["total_time"])
        args["dt"] = args["total_time"] / (args["num_timepoints"]-1)

        # If a pre-existing parameter file is to be used then load it.
        if args['upload_params']:

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



        items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]

        ################### PART THREE: SOLVE ######################
        timestamp = str(datetime.datetime.now())
        timestamp = timestamp.replace(':', '-')[:16]
        timestamp = timestamp.replace(' ', '_')
        print("Saving parameters...")
        with open(f"parameters.pkl", "wb") as file:
            pickle.dump(params_and_arrays, file)

        if len(items) % 10 == 0:
            chunks = [items[x:x+int(len(items)/10)] for x in range(0, len(items), int(len(items)/10))]
        else:
            chunks = [items]

        results_dict = dict()
        for i in chunks:
            print("Running solver...")
            results = multiprocess_wrapper(run_solver, i, args["jobs"])


            for d in results:
                for k, v in d.items():
                    results_dict[k] = v
            print("Saving results...")
            # Saving results
            with open(f"{timestamp}{args['growth']}_results.pkl", "wb") as file:
                pickle.dump(results_dict, file)


    # If you wish to do neighbourhood searching of previous results.
    if args['neighbourhood']:

        # Import prior results.
        # infile = open(args['results_file'], 'rb')
        # results_dict = pickle.load(infile)
        # infile.close()

        # Import parameters.
        infile = open(args['param_file'], 'rb')
        parameter_data = pickle.load(infile)
        infile.close()

        # Loop through results to identify hits.
        # hits = {}
        # for i in results_dict:
        #     if results_dict[i]['Fourier'][0] and results_dict[i]['Fourier'][1]:
        #         hits[i] = results_dict[i]

        for key, hit_params in parameter_data.items():

            ################### PART ONE: ATLAS ########################
            print("Extracting topology...")
            atlas = {0:hit_params[-1]}

            ################### PART TWO: PARAMETERS ###################
            print("Sampling parameters...")
            from parameters import LHS_neighbourhood

            nsamples = args['num_samples']
            
            hit_params = hit_params[0]
            hit_params.pop('alphan_x')
            hit_params.pop('alphan_y')

            sampler = LHS_neighbourhood(nsamples=args['num_samples'], params = hit_params)
            param_set = sampler.sample_parameters()

            # Prepare grid space, rate, and time
            args["J"] = args["system_length"]
            args["num_timepoints"] = int(12. * args["total_time"])
            args["dt"] = args["total_time"] / (args["num_timepoints"]-1)

            # Calculate alpha values for each species.
            for p in param_set:
                param_set[p]['alphan_x'] = Solver.calculate_alpha(param_set[p]['diffusion_x'], **args)
                param_set[p]['alphan_y'] = Solver.calculate_alpha(param_set[p]['diffusion_y'], **args)

            # Join altas and params
            indexes = product(param_set.keys(), atlas.keys())
            combinations = product(param_set.values(), atlas.values())

            params_and_arrays = {index: combination for index, combination in zip(indexes, combinations)}

            items = [(pa, params_and_arrays[pa], args) for pa in params_and_arrays]

            print("Saving parameters...")
            with open(f"neighbourhood/{key}_neighbourhood_parameters.pkl", "wb") as file:
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
                with open(f"neighbourhood/{key}_results.pkl", "wb") as file:
                    pickle.dump(results_dict, file)
    """
    def plot_conc(U):



        plt.plot(U)

        # fig, ax1 = plt.subplots()
        # color = 'tab:green'
        # ax1.set_xlabel('Space')
        # ax1.set_ylabel('Concentration x', color=color)
        # ax1.plot(U, color=color)
        # ax1.tick_params(axis='y')

        # ax2 = ax1.twinx()
        # color = 'tab:blue'
        # ax2.set_ylabel('Concentration y', color=color)
        # ax2.plot(U[1], color=color)
        # ax2.tick_params(axis='y')

        # fig.tight_layout()
        plt.show()

    print(results_dict.keys())
    for i in results_dict:
        print(results_dict[i]["steadystate"])
    conc = results_dict[(21963, 0, 1)]["concs"][0]
    print(len(conc))
    # print(conc)

    conc = [conc[i] for i in range(len(conc)) if i % 50 ==0]
    conc = conc[100:]

    # print(len(conc))
    c = np.array([np.format_float_positional(i, precision=8, unique=False, fractional=False, trim='k') for i in conc[-1]])

    plt.plot(np.linspace(0,49,50),c)
    plt.show()
    input()
    # input()

    stacked = np.vstack(conc)
    fig = plt.figure()
    axis = plt.axes(xlim=(0,49),ylim=(np.min(stacked),np.max(stacked)))
    line, = axis.plot([], [], lw = 3)

    def init():
        line.set_data([], [])
        return line

    def animate(i):
        x = np.array([i for i in range(50)])
        y = conc[i]
        line.set_data(x,y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(conc))
    anim.save('concWave.gif', fps = 240)
    """
