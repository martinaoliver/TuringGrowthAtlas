import numpy as np
import itertools, copy, sys
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, diags
from tqdm import tqdm
from scipy.linalg import solve_banded
import pandas as pd
np.random.seed(1)

class Solver: # Defines iterative solver methods

    def calculate_alpha(D, dt, dx):
        # Calculate alpha variable
        # D is the diffusion coefficient
        # dt is time step
        # dx is spatial step
        return D*dt/(2.*dx*dx)

    def a_matrix(alphan, size):
        print(size)
        print(alphan)

        # Function for preparing inverse A matrix for A^(-1).(B.U + f(u)
        # Diffusion terms for U+1 time step
        # alphan = D / (2 * delta_x**2) where D is the diffusion constant and delta_x is spatial grid separation
        bottomdiag = [-alphan for j in range(size-1)]
        centraldiag = [1.+alphan]+[1.+2.*alphan for j in range(size-2)]+[1.+alphan]
        topdiag = [-alphan for j in range(size-1)]
        diagonals = [bottomdiag,centraldiag,topdiag]
        A = diags(diagonals, [ -1, 0,1]).toarray()
        return np.linalg.inv(A)

    def b_matrix(alphan, size):
        # Function for preparing B matrix for A^(-1).(B.U + f(u)
        # Diffusion terms for U time step
        # alphan = D / (2 * delta_x**2) where D is the diffusion constant and delta_x is spatial grid separation
        bottomdiag = [alphan for j in range(size-1)]
        centraldiag = [1.-alphan]+[1.-2.*alphan for j in range(size-2)]+[1.-alphan]
        topdiag = [alphan for j in range(size-1)]
        diagonals = [bottomdiag,centraldiag,topdiag]
        B = diags(diagonals, [ -1, 0,1]).toarray()
        return B

    def hill_equations(interaction, rate):
        # Returns hill equation as a lambda function for a specified interaction

        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1 / (1 + (rate / concentration) ** 3) # This function is only formatted strangely to avoid an error.

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1 / (1 + (concentration / rate) ** 3)

    def create(size,perturbation=0.001,steadystate=0.1):
        # Create array concentration grid attribute
        perturbation = perturbation*steadystate
        low = steadystate - perturbation
        high = steadystate + perturbation
        return np.random.uniform(low=low, high=high, size=size)

    def grow(array):
        # Grow grid
        # Add row
        array = np.concatenate((array,[array[-1]]))

        if len(array.shape) == 2:
            # 2D growth
            end_column = np.array([array[:,-1]]).T
            # Add column
            array = np.concatenate((array,end_column), axis=1)

        return array

    def solve(p, topology, args):
        print(p)
        J = args["system_length"]
        dx = float(J)/(float(J)-1)

        total_time = args["total_time"]
        num_timepoints = 12*total_time
        dt = float(total_time) / float(num_timepoints - 1)

        # for param in ['diffusion_x', 'diffusion_y']:
        #
        #     # Calculate alpha values for each species.
        #     p[f"alphan_{param[-1]}"] = Solver.calculate_alpha(p[param], dx, dt)

        # Calculate A and B matrices for each species.
        A_matrix = [Solver.a_matrix(p["alphan_x"],J),Solver.a_matrix(p["alphan_y"],J)]
        B_matrix = [Solver.b_matrix(p["alphan_x"],J),Solver.b_matrix(p["alphan_y"],J)]
        print(A_matrix)
        print(len(A_matrix))

        # Create the reaction equations for this parameter set and topology.
        def react(conc):
            f_x = p['production_x'] - p['degradation_x']*conc[0] + p['max_conc_x'] * (Solver.hill_equations(topology[0,0], p['k_xx'])(conc[0])) * (Solver.hill_equations(topology[0,1], p['k_yx']))(conc[1])
            f_y = p['production_y'] - p['degradation_y']*conc[1] + p['max_conc_y'] * (Solver.hill_equations(topology[1,0], p['k_xy'])(conc[0])) * (Solver.hill_equations(topology[1,1], p['k_yy']))(conc[1])
            return np.array([f_x, f_y])

        # Set up starting conditions.

        currentJ = J
        ss = [4.54089711, 7.71675421]

        concentrations = [Solver.create(size = currentJ, steadystate=i) for i in ss]

        # Begin solving.

        for ti in range(num_timepoints):

            concentrations_new = copy.deepcopy(concentrations)

            reactions = react(concentrations)*dt
            concentrations_new = [np.dot(A_matrix[n], (B_matrix[n].dot(concentrations_new[n]) + reactions[n])) for n in range(2)]

            concentrations = copy.deepcopy(concentrations_new)

        return concentrations

    def plot_conc(U):
        plt.plot(U[0], label='U')
        plt.plot(U[1], label='V')
        plt.xlabel('Space')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()
