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

    def hill_equations(interaction, rate, n):
        # Returns hill equation as a lambda function for a specified interaction

        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1/(1 + (np.abs(rate/concentration))**n)

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1/(1 + (np.abs(concentration/rate))**n)

    def create(size,perturbation=0.001,steadystate=0.1):
        # Create array concentration grid attribute
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

        J = args["system_length"]
        dx = float(J)/(float(J)-1)

        total_time = args["total_time"]
        num_timepoints = 10*total_time
        dt = float(total_time) / float(num_timepoints - 1)

        for param in ['diffusion_x', 'diffusion_y']:
            
            # Calculate alpha values for each species.
            p[f"alphan_{param[-1]}"] = Solver.calculate_alpha(p[param], dx, dt)
                
        # Calculate A and B matrices for each species.
        if args["growth"] == None:
            A_matrices = [[Solver.a_matrix(p["alphan_x"],J),Solver.a_matrix(p["alphan_y"],J)]]
            B_matrices = [[Solver.b_matrix(p["alphan_x"],J),Solver.b_matrix(p["alphan_y"],J)]]
                
        # If growth is occurring, generate a list of A and B matrices for each new size.
        if args["growth"] == "linear":
            A_matrices = [[Solver.a_matrix(p["alphan_x"],j+1),Solver.a_matrix(p["alphan_y"],j+1)] for j in range(J)]
            B_matrices = [[Solver.b_matrix(p["alphan_x"],j+1),Solver.b_matrix(p["alphan_y"],j+1)] for j in range(J)]

        # Create the reaction equations for this parameter set and topology. 
        def react(conc):
            f_x = p['production_x'] - p['degradation_x']*conc[0] + p['max_conc_x'] * (Solver.hill_equations(topology[0,0], p['k_xx'], p['n'])(conc[0])) * (Solver.hill_equations(topology[0,1], p['k_yx'], p['n']))(conc[1])
            f_y = p['production_y'] - p['degradation_y']*conc[0] + p['max_conc_y'] * (Solver.hill_equations(topology[1,0], p['k_xy'], p['n'])(conc[0])) * (Solver.hill_equations(topology[1,1], p['k_yy'], p['n']))(conc[1])
            return np.array([f_x, f_y])

        # Set up starting conditions.

        A_matrix = A_matrices[0]
        B_matrix = B_matrices[0]

        if args['growth'] == None:
            currentJ = J
        elif args['growth'] == 'linear':
            currentJ = 1

        concentrations = [Solver.create(size = currentJ) for i in range((2))]

        # Begin solving.

        for ti in range(num_timepoints):

            concentrations_new = copy.deepcopy(concentrations)

            reactions = react(concentrations)*dt
            concentrations_new = [np.dot(A_matrix[n], (B_matrix[n].dot(concentrations_new[n]) + reactions[n])) for n in range(2)]

            hour = ti / num_timepoints / args['total_time']
            if args["growth"] == "linear" and hour % 1==0:
                concentrations_new = [Solver.grow(c) for c in concentrations_new]
                A_matrix = A_matrices[currentJ]
                B_matrix = B_matrices[currentJ]
                currentJ +=1

            concentrations = copy.deepcopy(concentrations_new)
            
        return concentrations
            
    def plot_conc(U):
        plt.plot(U[0], label='U')
        plt.plot(U[1], label='V')
        plt.xlabel('Space')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()

