# Author Xindong
# Date 2022/2/9 15:51
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

np.random.seed(1)

class NewtonRaphson:
    # Defines Newton Raphson methods for finding steadystates

    def initiate_jacobian(params, hill):
        # Generate Jacobian matrix
        # Retains expressions with X and Y as placeholder symbols
        X, Y = symbols('X'), symbols('Y')
        functions = Matrix(Solver.react([X, Y], params, **hill))
        jacobian_topology = functions.jacobian([X, Y])
        return jacobian_topology, X, Y, functions

    def iterate(x_initial, params, hill, jac, X, Y, max_num_iter=30, tolerance=0.0001, alpha=1):
        # Perform NR iteration on one initial condition
        # Max number of iteration is 15 by default
        x = x_initial
        fx = Solver.react(x, params, **hill)
        err = np.linalg.norm(fx)
        iter = 0

        # perform the Newton-Raphson iteration
        while err > tolerance and iter < max_num_iter and np.all(x != 0):

            jac_temp = jac.subs(X, x[0])
            jac_temp = jac_temp.subs(Y, x[1])
            jac_temp = np.array(jac_temp, dtype=float)

            # update
            step = alpha * np.linalg.solve(jac_temp, fx)
            x = x - step
            fx = Solver.react(x, params, **hill)
            err = np.linalg.norm(fx)
            iter += 1

        # check that there are no negatives
        if err < tolerance:
            if sum(item < 0 for item in x) == 0:
                return (x, err, 0)

    def run(initial_conditions, params, hill):
        # Run Newton Raphson algorithm on multiple conditions
        # If steady state identified add to steady state list
        # Returns list of steady states

        SteadyState_list = []

        # initialise jacobian matrix
        jac, X, Y, functions = NewtonRaphson.initiate_jacobian(params, hill)

        # loop through sampled conditions
        for condition in initial_conditions:
            # Perform newton raphson iteration on one initial condition
            xn = NewtonRaphson.iterate(condition, params, hill, jac, X, Y)

            # If xn is returned as a list check for duplicates in SteadyState_list,
            # as well as complex numbers and negative numbers
            if xn:
                # Retrieve concentration arrays from xn
                xs = xn[0]

                # Check for negative instances
                if np.all(xs > 0):
                    if len(SteadyState_list) == 0:
                        SteadyState_list.append(xs)
                    else:
                        # compare with previous steady states for duplicates
                        logiclist = []
                        for state in SteadyState_list:
                            logiclist.append(
                                np.allclose(state, xs, rtol=10 ** -2,atol=0))  # PROCEED IF NO TRUES FOUND
                        if not True in logiclist:  # no similar steady states previously found
                            SteadyState_list.append(xs)

        return SteadyState_list


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


    def loguniform(low=-3, high=3, size=None):
        # Return a logarithmic distribution
        return (10) ** (np.random.uniform(low, high, size))

    def lhs_list(data, nsample):
        # This function accepts an already existing distribution and a required number of samples
        # and outputs an array with samples distributed in a latin-hyper-cube sampling manner.
        nvar = data.shape[1]
        ran = np.random.uniform(size=(nsample, nvar))
        s = np.zeros((nsample, nvar))
        for j in range(0, nvar):
            idx = np.random.permutation(nsample) + 1
            P = ((idx - ran[:, j]) / nsample) * 100
            s[:, j] = np.percentile(data[:, j], P)
        return s

    def lhs_initial_conditions(n_initialconditions=10, n_species=2):
        # Input number of initial conditions needed and the number of species in each sample obtain an array with the
        # initial conditions distributed in a lhs manner.
        data = np.column_stack(([Solver.loguniform(size=100000)] * n_species))
        initial_conditions = Solver.lhs_list(data, n_initialconditions)
        return np.array(initial_conditions, dtype=np.float)

    def create(steadystate, size, perturbation=0.001):
        # define the initial value from steady state
        low = steadystate - perturbation
        high = steadystate + perturbation
        return np.random.uniform(low=low, high=high, size=size)


    def exponential_growth(initial_dx,dt,t,s=0.0001):
        return initial_dx*np.exp(s*t)

    def linear_growth(initial_dx,dt,t,s=0.00005):
        return initial_dx + t*s

    def weird_growth(t,dt):
        s=0.01
        tf=1000
        return 40*(2+np.tanh(s*(t-tf/2)))

    def hill_equations(interaction, rate):
        # Returns hill equation as a lambda function for a specified interaction

        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1 / (1 + (rate / concentration) ** 2)

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1 / (1 + (concentration / rate) ** 2)

    def react(conc, params, hillxx, hillxy, hillyx, hillyy):
        # Function for performing one f(u,v) step
        X, Y = conc
        fx = params['production_x'] - params['degradation_x'] * X + params['max_conc_x'] * hillxx(X) * hillyx(Y)
        fy = params['production_y'] - params['degradation_y'] * Y + params['max_conc_y'] * hillxy(X) * hillyy(Y)

        return np.array([fx, fy])

    def solve(params, topology, growth, dt, dx, J, total_time, num_timepoints, **kwargs):

        # Calculate diffussion matrices
        if growth:
            A_matrices = [[Solver.a_matrix(params["alphan_x"][i], J), Solver.a_matrix(params["alphan_y"][i], J)] for i in range(num_timepoints)]
            B_matrices = [[Solver.b_matrix(params["alphan_x"][i], J), Solver.b_matrix(params["alphan_y"][i], J)] for i in range(num_timepoints)]

        else:
            A_matrices = [[Solver.a_matrix(params["alphan_x"][0], J), Solver.a_matrix(params["alphan_y"][0], J)]]
            B_matrices = [[Solver.b_matrix(params["alphan_x"][0], J), Solver.b_matrix(params["alphan_y"][0], J)]]


        # Calculate A and B matrices for each species.
        # Define hill equations
        hill = dict(
            hillxx = Solver.hill_equations(topology[0, 0], params['k_xx']),
            hillyx = Solver.hill_equations(topology[0, 1], params['k_yx']),
            hillxy = Solver.hill_equations(topology[1, 0], params['k_xy']),
            hillyy = Solver.hill_equations(topology[1, 1], params['k_yy'])
        )

        # find the steady state
        initial_conditions = Solver.lhs_initial_conditions(n_initialconditions=100, n_species=2)
        SteadyState_list = NewtonRaphson.run(initial_conditions, params, hill)

        if SteadyState_list:
            conc_list = []

            for steady_conc in SteadyState_list:
                A_matrix = A_matrices[0]
                B_matrix = B_matrices[0]

                currentJ = J

                concentrations = [Solver.create(steady_conc[i], size=currentJ) for i in range(2)]

                for ti in range(num_timepoints):
                    concentrations_new = copy.deepcopy(concentrations)

                    reactions = Solver.react(concentrations, params, **hill) * dt
                    concentrations_new = [np.dot(A_matrix[n], (B_matrix[n].dot(concentrations_new[n]) + reactions[n]))
                                          for n in range(2)]

                    concentrations = copy.deepcopy(concentrations_new)

                turing = Solver.fourier_classify(concentrations)

                conc_list.append((concentrations, turing))

            return conc_list, SteadyState_list

        else:
            return [None], [None]

    def plot_conc(U):
        plt.plot(U[0], label='U')
        plt.plot(U[1], label='V')
        plt.xlabel('Space')
        plt.ylabel('Concentration')
        plt.legend()
        plt.show()

    def fourier_classify(U, threshold = 2, plot = False):

        # Compute the fourier transforms.
        transforms = [fft(i) for i in U]

        # Check for peaks.
        peaks_found = False

        for i in transforms:
            for ii in i[1:]:
                if abs(ii) > threshold:
                    peaks_found = True

        # Plot the fourier transforms.
        # if plot:
        #     for i in [0,1]:
        #         freq = fftfreq(L,dx)
        #         plt.plot(freq, abs(transforms[i]))
        #         plt.xlim(0,)
        #         plt.show()

        return peaks_found
