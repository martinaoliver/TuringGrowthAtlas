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

class NewtonRaphson:
    # Defines Newton Raphson methods for finding steadystates

    def initiate_jacobian(params, hill):
        # Generate Jacobian matrix
        # Retains expressions with X and Y as placeholder symbols
        X, Y = symbols('X'), symbols('Y')
        functions = Matrix(Solver.react([X, Y], params, **hill))
        jacobian_topology = functions.jacobian([X, Y])
        return jacobian_topology, X, Y

    def iterate(x_initial, params, hill, jac, X, Y, max_num_iter=15, tolerance=0.0001, alpha=1):
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
            x = x - alpha * np.linalg.solve(jac_temp, fx)
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
        jac, X, Y = NewtonRaphson.initiate_jacobian(params, hill)

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

    def create(steadystate, size, growth, initial, perturbation=0.001,):
        # define the initial value from steady state
        low = steadystate - perturbation
        high = steadystate + perturbation
        conc = np.random.uniform(low=low, high=high, size=size)
        if not growth:
            return conc
        if growth:
            conc = np.multiply(conc,initial)
            return conc


    def exponential_growth(t, s=0.0001, initialL=1):
        return (initialL*np.exp(s*t))

    def linear_growth(t,s=0.051, initialL=1):
        return initialL + t*s

    def growth_bounds(concs, boul_array, num_cells):
        cells_to_add = num_cells - np.count_nonzero(boul_array)
        concs = [np.multiply(conc_array,boul_array) for conc_array in concs]
        full = np.where(boul_array == 1)[0]
        start = int(len(boul_array)/2)

        for cell in range(cells_to_add):
            if abs(full[-1]-start) >= abs(full[0]-start):
                loc = False
            else:
                loc = True

            if np.all(boul_array==1):
                break
            if boul_array[-1] == 1:
                boul_array[full[0]-1] = 1
                for conc_array in concs:
                    conc_array[full[0]-1] = conc_array[full[0]]

            elif boul_array[0] == 1:
                boul_array[full[-1]+1] = 1
                for conc_array in concs:
                    conc_array[full[-1]+1] = conc_array[full[-1]]

            elif loc:
                boul_array[full[-1]+1] = 1
                for conc_array in concs:
                    conc_array[full[-1]+1] = conc_array[full[-1]]

            else:
                boul_array[full[0]-1] = 1
                for conc_array in concs:
                    conc_array[full[0]-1] = conc_array[full[0]]

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

    # Define LSA check functions
    def diffusion_jacobian(params, hill, steady_state):
        # Generate diffusion Jacobian matrix with steady state
        X, Y = symbols('X'), symbols('Y')
        functions = Matrix(Solver.react([X, Y], params, **hill))
        jac = functions.jacobian([X, Y])
        # Substitute X, Y with steady state value
        Xs, Ys = steady_state
        jac_diff = jac.subs(X, Xs)
        jac_diff = jac_diff.subs(Y, Ys)
        jac_diff = np.array(jac_diff, dtype=float)

        return jac_diff

    def calculate_dispersion(params, hill, steady_state, top_dispersion=5000, n_species=2):
        jac_diff = Solver.diffusion_jacobian(params, hill, steady_state)
        wvn_list = np.array(list(range(0, top_dispersion + 1))) * np.pi / 100
        count = 0
        eigenvalues = np.zeros((len(wvn_list), n_species), dtype=np.complex)
        Diff_matrix = np.array([[params['diffusion_x'], 0], [0, params['diffusion_y']]])

        for wvn in wvn_list:
            jac_temp = jac_diff - Diff_matrix * wvn**2 # calculate the diffusion jacobian matrix
            eigenval, eigenvec = np.linalg.eig(jac_temp) # solve the eigenvalue
            eigenvalues[count] = np.sort(eigenval) # sort eigenvalues
            count += 1

        return eigenvalues

    # Independent LSA solver
    def LSA(params, topology, hill, steady_conc):
        # LSA check
        turing = None # 0 for no typical turing, 1 for typical turing
        K = None
        eigen_v = Solver.calculate_dispersion(params, hill, steady_conc) # calculate the eigenvalue
        eigen_v_min = eigen_v[:,1] # take the maximum eigenvalue, second column
        eigen_min_r = eigen_v_min.real # take the real part
        if eigen_min_r[0] < 0 and eigen_min_r[-1] < 0: # check head and tail
            if np.max(eigen_min_r) > 0: # check the middle
                turing = 1
                K = np.argmax(eigen_min_r) * np.pi / 100 # find the wavenumber of maximum eigenvalue

        return [turing,K]

    def solve(params, topology, growth, dt, dx, J, total_time, num_timepoints, **kwargs):

        # Calculate A and B matrices for each species.
        A_matrices = [Solver.a_matrix(params["alphan_x"], J), Solver.a_matrix(params["alphan_y"], J)]
        B_matrices = [Solver.b_matrix(params["alphan_x"], J), Solver.b_matrix(params["alphan_y"], J)]

        # Define hill equations
        hill = dict(
            hillxx = Solver.hill_equations(topology[0, 0], params['k_xx']),
            hillyx = Solver.hill_equations(topology[0, 1], params['k_yx']),
            hillxy = Solver.hill_equations(topology[1, 0], params['k_xy']),
            hillyy = Solver.hill_equations(topology[1, 1], params['k_yy'])
        )

        # Find the steady state
        initial_conditions = Solver.lhs_initial_conditions(n_initialconditions=100, n_species=2)
        SteadyState_list = NewtonRaphson.run(initial_conditions, params, hill)


        # Set up starting conditions.

        # Begin solving.
        if SteadyState_list:
            conc_list = []
            LSA_list = []
            fourier_list = []

            for steady_conc in SteadyState_list:

                if not growth:
                    LSA_list.append(Solver.LSA(params, topology, hill, steady_conc))
                else:
                    LSA_list.append(None)
                # bool_list = list()
                # Crank Nicolson solver
                # bool array used only for growth
                bool_array = np.zeros(J)
                bool_array[int(J/2)] = 1
                # bool_list.append(bool_array)
                concentrations = [Solver.create(steady_conc[i], size=J, growth=growth, initial=bool_array) for i in range(2)]
                newL = 1
                ccc = [concentrations]
                for ti in range(num_timepoints):
                    # Extra steps to prevent division by 0 when calculating reactions
                    concentrations_new = copy.deepcopy(concentrations)
                    full = np.where(concentrations_new[0]!=0)[0]
                    concs_react = [conc[conc!=0] for conc in concentrations_new]
                    reactions = Solver.react(concs_react, params, **hill) * dt
                    reactions_padded = copy.deepcopy(concentrations_new)
                    for i in range(len(reactions_padded)):
                        reactions_padded[i][full] = reactions[i]
                    concentrations_new = [np.dot(A_matrices[n], (B_matrices[n].dot(concentrations_new[n]) + reactions_padded[n]))
                                          for n in range(2)]

                    hour = ti / (num_timepoints / total_time)
                    if growth == 'exponential':
                        if newL < J:
                            newL = int(Solver.exponential_growth(hour))
                            concentrations_new = Solver.growth_bounds(concentrations_new, bool_array, newL)


                    if growth == 'linear':
                        if newL < J:
                            newL = int(Solver.linear_growth(hour))
                            concentrations_new, bool_array = Solver.growth_bounds(concentrations_new, bool_array, newL)

                    concentrations = copy.deepcopy(concentrations_new)
                    ccc.append(concentrations[0])

                fourier = Solver.fourier_classify(concentrations)
                peaks = Solver.peaks_classify(concentrations)
                if fourier and peaks:
                    print('Found one!')
                # t_grid = np.array([n*dt for n in range(num_timepoints+2)])
                # x_grid = np.array([n*dx for n in range(50)])
                # Solver.surfpattern(ccc, grids=[x_grid,t_grid])
                # Solver.plot_conc(concentrations)
                fourier_list.append((fourier,peaks))
                conc_list.append(concentrations)


            return conc_list, SteadyState_list, LSA_list, fourier_list

        else:
            return [None], [None], [None], [None]

    def plot_conc(U):

        fig, ax1 = plt.subplots()
        color = 'tab:green'
        ax1.set_xlabel('Space')
        ax1.set_ylabel('Concentration x', color = color)
        ax1.plot(U[0], color = color)
        ax1.tick_params(axis = 'y')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Concentration y', color = color)
        ax2.plot(U[1], color = color)
        ax2.tick_params(axis='y')

        fig.tight_layout()
        plt.show()

    def surfpattern(results,grids,morphogen = 0):
        results = np.vstack(results)
        results = np.transpose(results)
        r_non_zero = results[results != 0]
        levels = [0.]
        for i in range(10):
            levels.append(np.percentile(r_non_zero, 10*i, interpolation='midpoint'))
        levels = list(dict.fromkeys(levels))
        x_grid = grids[0]
        t_grid = grids[1]
        t,x = np.meshgrid(t_grid,x_grid)
        plt.contourf(x,t,results, cmap=cmap,levels=levels)
        plt.colorbar()
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.show()

    def fourier_classify(U, threshold = 2, plot = False):

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
             for i in [0,1]:
                 freq = fftfreq(50,0.3)
                 plt.plot(freq, abs(transforms[i]))
                 plt.xlim(0,)
                 plt.show()

        return peaks_found

    def peaks_classify(U):
        peaks = [len(find_peaks(i)[0]) for i in U]
        multiple_peaks_found = False
        for i in peaks:
            if i > 2:
                multiple_peaks_found = True

        return multiple_peaks_found
