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

np.random.seed(1)

class NewtonRaphson:
    # Defines Newton Raphson methods for finding steadystates

    def initiate_jacobian():
        X, Y = symbols('X'), symbols('Y')
        arguments = Matrix([X, Y])
        functions = Matrix(react([X, Y]))
        jacobian_topology = functions.jacobian(arguments)
        return jacobian_topology

    def iterate(x_initial, max_num_iter=15, tolerance=0.0001, alpha=1):
        x = x_initial
        fx = react(x)
        err = np.linalg.norm(fx)
        iter = 0
        jac = jacobian1()
        # print(jac)
        X, Y = symbols('X'), symbols('Y')

        # perform the Newton-Raphson iteration
        while err > tolerance and iter < max_num_iter and np.all(x != 0):

            jac_temp = jac.subs(X, x[0])
            jac_temp = jac_temp.subs(Y, x[1])
            # print(jac)
            jac_temp = np.array(jac_temp, dtype=float)

            # update
            x = x - alpha * np.linalg.solve(jac_temp, fx)
            fx = react(x)
            err = np.linalg.norm(fx)
            iter += 1

        # check that there are no negatives
        if err < tolerance:
            if sum(item < 0 for item in x) == 0:
                return (x, err, 0)

    def run(initial_conditions):
        count = 0
        SteadyState_list = []
        for n in range(len(initial_conditions)):
            xn = []
            xn = newton_raphson(initial_conditions[n])

            if xn != None:
                if count == 0:
                    SteadyState_list.append(xn[0])
                    count += 1
                if count > 0:  # repeats check: compare with previous steady states
                    logiclist = []
                    for i in range(count):
                        logiclist.append(
                            np.allclose(SteadyState_list[i], xn[0], rtol=10 ** -2,
                                        atol=0))  # PROCEED IF NO TRUES FOUND
                    if not True in logiclist:  # no similar steady states previously found
                        SteadyState_list.append(xn[0])
                        count += 1

        return SteadyState_list


class Solver:  # Defines iterative solver methods

    def calculate_alpha(D, dt, dx):
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
        return (10) ** (np.random.uniform(low, high, size))

    def lhs_list(data, nsample):
        nvar = data.shape[1]
        ran = np.random.uniform(size=(nsample, nvar))
        s = np.zeros((nsample, nvar))
        for j in range(0, nvar):
            idx = np.random.permutation(nsample) + 1
            P = ((idx - ran[:, j]) / nsample) * 100
            s[:, j] = np.percentile(data[:, j], P)
        return s

    def lhs_initial_conditions(n_initialconditions=10, n_species=2):
        data = np.column_stack(([Solver.loguniform(size=100000)] * n_species))
        initial_conditions = Solver.lhs_list(data, n_initialconditions)
        return np.array(initial_conditions, dtype=np.float)

    # define the initial value from steady state
    def create(steadystate, size, perturbation=0.001):
        low = steadystate - perturbation
        high = steadystate + perturbation
        return np.random.uniform(low=low, high=high, size=size)

    def grow(array):
        # Grow grid
        # Add row
        array = np.concatenate((array, [array[-1]]))

        if len(array.shape) == 2:
            # 2D growth
            end_column = np.array([array[:, -1]]).T
            # Add column
            array = np.concatenate((array, end_column), axis=1)

        return array

    def hill_equations(interaction, rate, n):
        # Returns hill equation as a lambda function for a specified interaction

        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1 / (1 + (rate / concentration) ** n))

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1 / (1 + (concentration / rate) ** n))

    def react(conc, p, hillxx, hillxy, hillyx, hillyy):
        # Function for performing one f(u,v) step
        X, Y = conc
        fx = p['production_x'] - p['degradation_x'] * X + p['max_conc_x'] * hillxx(X) * hillyx(Y)
        fy = p['production_y'] - p['degradation_y'] * Y + p['max_conc_y'] * hillxy(X) * hillyy(Y)

        return np.array([fx, fy])

    def solve(p, topology, args):

        J = args["system_length"]
        dx = float(J) / (float(J) - 1)

        total_time = args["total_time"]
        num_timepoints = 10 * total_time
        dt = float(total_time) / float(num_timepoints - 1)

        for param in ['diffusion_x', 'diffusion_y']:
            # Calculate alpha values for each species.
            p[f"alphan_{param[-1]}"] = Solver.calculate_alpha(p[param], dx, dt)

        # Calculate A and B matrices for each species.
        if args["growth"] == None:
            A_matrices = [[Solver.a_matrix(p["alphan_x"], J), Solver.a_matrix(p["alphan_y"], J)]]
            B_matrices = [[Solver.b_matrix(p["alphan_x"], J), Solver.b_matrix(p["alphan_y"], J)]]

        # If growth is occurring, generate a list of A and B matrices for each new size.
        if args["growth"] == "linear":
            A_matrices = [[Solver.a_matrix(p["alphan_x"], j + 1), Solver.a_matrix(p["alphan_y"], j + 1)] for j in
                          range(J)]
            B_matrices = [[Solver.b_matrix(p["alphan_x"], j + 1), Solver.b_matrix(p["alphan_y"], j + 1)] for j in
                          range(J)]

        # Define hill equations
        hill = dict(
            hillxx = Solver.hill_equations(topology[0, 0], params['k_xx'], params['n_xx']),
            hillyx = Solver.hill_equations(topology[0, 1], params['k_yx'], params['n_yx']),
            hillxy = Solver.hill_equations(topology[1, 0], params['k_xy'], params['n_xy']),
            hillyy = Solver.hill_equations(topology[1, 1], params['k_yy'], params['n_yy'])
        )


        # find the steady state
        initial_conditions1 = Solver.lhs_initial_conditions(n_initialconditions=100, n_species=2)
        SteadyState_list = newtonraphson_run(initial_conditions1)
        print("Steady Done")

        # Set up starting conditions.
        A_matrix = A_matrices[0]
        B_matrix = B_matrices[0]

        if args['growth'] == None:
            currentJ = J
        elif args['growth'] == 'linear':
            currentJ = 1

        # Begin solving.
        if SteadyState_list:
            conc_list = []
            for steady_conc in SteadyState_list:
                concentrations = [Solver.create(steady_conc[i], size=currentJ) for i in range(2)]

                for ti in range(num_timepoints):

                    concentrations_new = copy.deepcopy(concentrations)

                    reactions = react(concentrations) * dt
                    concentrations_new = [np.dot(A_matrix[n], (B_matrix[n].dot(concentrations_new[n]) + reactions[n]))
                                          for n in range(2)]

                    hour = ti / num_timepoints / args['total_time']
                    if args["growth"] == "linear" and hour % 1 == 0:
                        concentrations_new = [Solver.grow(c) for c in concentrations_new]
                        A_matrix = A_matrices[currentJ]
                        B_matrix = B_matrices[currentJ]
                        currentJ += 1

                    concentrations = copy.deepcopy(concentrations_new)

                conc_list.append(concentrations)

            print("Solved")
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
