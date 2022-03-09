from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import itertools, copy, sys
from scipy.sparse import spdiags, diags
from scipy.linalg import solve_banded
from scipy.special import logsumexp
from scipy import optimize
from sympy import *
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
import random
from matplotlib import cm

cmap = cm.Spectral

np.random.seed(1)


class Solver:  # Defines iterative solver methods

    def calculate_alpha(D, dt, dx, **kwargs):
        # Calculate alpha variable
        # D is the diffusion coefficient
        # dt is time step
        # dx is spatial step
        return D * dt / (2. * dx * dx)

    def a_matrix(alphan, size):
        # Function for preparing inverse A matrix for A^(-1).(B.U + f(u)
        # Diffusion terms for U+1 time step
        # alphan = D / (2 * delta_x**2) where D is the diffusion constant and delta_x is spatial grid separation
        bottomdiag = [-alphan for j in range(size - 1)]
        centraldiag = [1. + alphan] + [1. + 2. * alphan for j in range(size - 2)] + [1. + alphan]
        topdiag = [-alphan for j in range(size - 1)]
        diagonals = [bottomdiag, centraldiag, topdiag]
        A = diags(diagonals, [-1, 0, 1]).toarray()
        return np.linalg.inv(A)

    def b_matrix(alphan, size):
        # Function for preparing B matrix for A^(-1).(B.U + f(u)
        # Diffusion terms for U time step
        # alphan = D / (2 * delta_x**2) where D is the diffusion constant and delta_x is spatial grid separation
        bottomdiag = [alphan for j in range(size - 1)]
        centraldiag = [1. - alphan] + [1. - 2. * alphan for j in range(size - 2)] + [1. - alphan]
        topdiag = [alphan for j in range(size - 1)]
        diagonals = [bottomdiag, centraldiag, topdiag]
        B = diags(diagonals, [-1, 0, 1]).toarray()
        return B


    def create(steadystate, size, growth, initial, perturbation=0.000001):
        # define the initial value from steady state
        perturbation = perturbation*steadystate
        low = steadystate - perturbation
        high = steadystate + perturbation
        conc = np.random.uniform(low=low, high=high, size=size)
        if not growth:
            return conc
        if growth:
            conc = np.multiply(conc, initial)
            return conc

    def exponential_growth(t, s=0.0001, initialL=2):
        return (initialL * np.exp(s * t))

    def linear_growth(t, s=0.03335, initialL=2):
        return initialL + t * s

    def growth_bounds(concs, boul_array):

        full = np.where(boul_array == 1)[0]
        for c in concs:
            c[full[0] - 1] = c[full[0]]
            c[full[-1] + 1] = c[full[-1]]
        boul_array[full[0] - 1] = 1
        boul_array[full[-1] + 1] = 1

        return concs, boul_array

    def hill_equations(interaction, rate):
        # Returns hill equation as a lambda function for a specified interaction

        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1 / (1 + (rate / concentration) ** 3)

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1 / (1 + (concentration / rate) ** 3)

    def react(conc, params, hillxx, hillxy, hillyx, hillyy):
        # Function for performing one f(u,v) step
        X, Y = conc
        fx = params['production_x'] - params['degradation_x'] * X + params['max_conc_x'] * hillxx(X) * hillyx(Y)
        fy = params['production_y'] - params['degradation_y'] * Y + params['max_conc_y'] * hillxy(X) * hillyy(Y)
        return np.array([fx, fy])

    def solve(params, topology, growth, dt, dx, J, total_time, num_timepoints, **kwargs):
        # Calculate A and B matrices for each species.
        print(params)
        A_matrices = [Solver.a_matrix(params["alphan_x"], J), Solver.a_matrix(params["alphan_y"], J)]
        B_matrices = [Solver.b_matrix(params["alphan_x"], J), Solver.b_matrix(params["alphan_y"], J)]
        print(A_matrices)
        print(len(A_matrices))
        # Define hill equations
        hill = dict(
            hillxx=Solver.hill_equations(topology[0, 0], params['k_xx']),
            hillyx=Solver.hill_equations(topology[0, 1], params['k_yx']),
            hillxy=Solver.hill_equations(topology[1, 0], params['k_xy']),
            hillyy=Solver.hill_equations(topology[1, 1], params['k_yy'])
        )


        SteadyState_list =[[4.54089711, 7.71675421]]

        # Set up starting conditions.

        # Begin solving.
        if SteadyState_list:
            conc_list = []
            LSA_list = []
            fourier_list = []

            for steady_conc in SteadyState_list:
                all_concs=[]

                bool_array = np.zeros(J)
                bool_array[int(J / 2)] = 1
                bool_array[int(J / 2) -1 ] = 1
                # bool_list.append(bool_array)
                concentrations = [Solver.create(steady_conc[i], size=J, growth=growth, initial=bool_array) for i in
                                  range(2)]

                newL = 2
                oldL = 2

                all_concs = [[concentrations[0]],[concentrations[1]]]

                for ti in range(num_timepoints):
                    # Extra steps to prevent division by 0 when calculating reactions
                    concentrations_new = copy.deepcopy(concentrations)

                    reactions = Solver.react(concentrations_new, params, **hill) * dt
                    concentrations_new = [
                        np.dot(A_matrices[n], (B_matrices[n].dot(concentrations_new[n]) + reactions[n]))
                        for n in range(2)]



                    concentrations = copy.deepcopy(concentrations_new)
                    if kwargs["save_all"]:
                        all_concs[0].append(concentrations[0])
                        all_concs[1].append(concentrations[1])
                fourier = Solver.fourier_classify(concentrations)
                peaks = Solver.peaks_classify(concentrations)
                if fourier and peaks:
                    print('Found one!')

                fourier_list.append((fourier, peaks))
                if not kwargs["save_all"]:
                    conc_list.append(concentrations)
                else:
                    conc_list.append(all_concs)

            return conc_list, SteadyState_list, LSA_list, fourier_list

        else:
            return [None], [None], [None], [None, None]

    def plot_conc(U):

        fig, ax1 = plt.subplots()
        color = 'tab:green'
        ax1.set_xlabel('Space')
        ax1.set_ylabel('Concentration x', color=color)
        ax1.plot(U[0], color=color)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Concentration y', color=color)
        ax2.plot(U[1], color=color)
        ax2.tick_params(axis='y')

        fig.tight_layout()
        plt.show()

    def surfpattern(results, grids, morphogen=0):
        results = np.vstack(results)
        results = np.transpose(results)
        r_non_zero = results[results != 0]
        levels = [0.]
        for i in range(10):
            levels.append(np.percentile(r_non_zero, 10 * i, interpolation='midpoint'))
        levels = list(dict.fromkeys(levels))
        x_grid = grids[0]
        t_grid = grids[1]
        t, x = np.meshgrid(t_grid, x_grid)
        plt.contourf(x, t, results, cmap=cmap, levels=levels)
        plt.colorbar()
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.show()

    def fourier_classify(U, threshold=2, plot=False):

        # Round to 5sf avoid picking up noise.
        U = [np.array([np.format_float_positional(conc, precision=5, unique=False, fractional=False, trim='k') for conc in morphogen]) for morphogen in U]

        # Compute the fourier transforms.
        transforms = [fft(i) for i in U]

        # Check for peaks.
        x_peaks = [abs(i) > threshold for i in transforms[0][1:]]
        y_peaks = [abs(i) > threshold for i in transforms[1][1:]]

        # Peaks must be found in both species to return True.
        peaks_found = False
        if sum(x_peaks) > 0 and sum(y_peaks) > 0:
            peaks_found = True

        # Plot the fourier transforms.
        if plot:
            for i in [0, 1]:
                freq = fftfreq(50, 0.3)
                plt.plot(freq, abs(transforms[i]))
                plt.xlim(0, )
                plt.show()

        return peaks_found

    def peaks_classify(U):
        peaks = [len(find_peaks(i)[0]) for i in U]
        multiple_peaks_found = False
        for i in peaks:
            if i > 2:
                multiple_peaks_found = True

        return multiple_peaks_found
