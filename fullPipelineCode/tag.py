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

class Grid: # Defines spatial concentration grid for one species

    def create(array=None,dimensions=(1,),perturbation=0.001,steadystate=0.1):
        # Create array concentration grid attribute
        if array == None:
            low = steadystate - perturbation
            high=steadystate + perturbation
            array = np.random.uniform(low=low, high=high, size=dimensions)
            return array
        else:
            return array

    def growth(array):
        # Grow grid
        # Add row
        array = np.concatenate((array,[array[-1]]))

        if len(array.shape) == 2:
            # 2D growth
            end_column = np.array([array[:,-1]]).T
            # Add column
            array = np.concatenate((array,end_column), axis=1)

        return array

class Topologies: # Defines morphogen interaction topologies

    def is_connected(self, M): #This function checks if a network M is connected (If so, it returns True)
        M1= M.copy() #create copy so initial array does not get modified
        np.fill_diagonal(M1, 0) #connections to itself do not count when identifying if a network is unconnected. Therefore we set those to zero (diagonal).

        ##Every column and row must have at least one non-zero value for the network to be connected.
        if all(M1.any(axis=0)):
            if all(M1.any(axis=1)):
                return True

    def remove_unconnected(self, matrix_array): #This function loops through the array of networks and deletes the UNCONNECTED networks.
        new_matrix_array = []
        for index,M in enumerate(matrix_array): #loop through matrices
            if self.is_connected(M)==True: #if matrix connected, add to new array
                new_matrix_array.append(M)
        return new_matrix_array

    def isomorphic(self, AH, diffusable=2): #This function returns list of isomorisms of graph AH.
        N=AH.shape[0]
        if N==2:
            P_array = [np.matrix('0 1; 1 0'),np.matrix('1 0; 0 1')]

        if N==3:
            if diffusable==2:
                P_array = [np.matrix('0 1 0; 1 0 0; 0 0 1')]
            elif diffusable==3:
                P_array = [
                np.matrix('1 0 0; 0 0 1; 0 1 0'), np.matrix('0 1 0; 1 0 0; 0 0 1'),
                np.matrix('0 1 0; 0 0 1; 1 0 0'), np.matrix('0 0 1; 0 1 0; 1 0 0'),
                np.matrix('0 0 1; 1 0 0; 0 1 0'), np.matrix('1 0 0; 0 1 0; 0 0 1')
            ]
        # Get transpose matrices for each.
        PT_array = [np.transpose(P) for P in P_array]
        P_arrays = zip(P_array, PT_array)

        # Get all the permutations of matrix A and sort them.
        permutations = [np.dot(np.dot(P,AH),PT) for (P,PT) in P_arrays]
        permutations = list(map(str,permutations))
        permutations.sort()
        permutations = ''.join(permutations)
        return permutations

    def remove_isomorphism(self, topologies):
        unique_topologies = {} # Dictionary to be populated with unique matrices.

        with tqdm(total = len(topologies)) as pbar:

            # For each netwoek, get permutations of that network.
            # Check if that set of permutations is already a key and if not, add them.

            for network in topologies:
                permutations = self.isomorphic(network)
                if permutations not in unique_topologies:
                    unique_topologies[permutations] = network
                pbar.update(1)

        return unique_topologies.values()

    def create_adjacency_matrices(self, num_nodes, diffusable=2):
        adj_matrix_array = [np.reshape(np.array(i), (num_nodes,num_nodes)) for i in itertools.product([0, 1,-1], repeat = num_nodes*num_nodes)] #array with all possible ajdacency matrices with an activation (1), inhibition (-1) or no connection (0).
        adj_matrix_array_connected = self.remove_unconnected(adj_matrix_array)
        atlas_matrix_array = self.remove_isomorphism(adj_matrix_array_connected)
        return atlas_matrix_array

class LHS: # Defines parameter sampling

    def __init__(self, n_samples = 200, species=2, params={}):
        letters = ['x','y','z']
        self.params = {}
        self.n_samples = n_samples
        for i in range(species):
            self.params[f'production_{letters[i]}'] = [0.1, 100]
            self.params[f'degradation_{letters[i]}'] = [0.01,1]
            self.params[f'max_conc_{letters[i]}'] = [0.1, 100]
            self.params[f'diffusion_{letters[i]}'] = [0.001, 1000]
            for j in range(species):
                if f'k_{letters[i]}{letters[j]}' not in self.params:
                    self.params[f'k_{letters[i]}{letters[j]}'] = [0.001, 1000]
        for key, value in params:
            self.params[key] = value

    # Function generates value from log distribution (0.001 to 1000).
    def loguniform(self,low=-3, high=3, size=None):
        return (10) ** (np.random.uniform(low, high, size))

    # Function generates non-overlapping samples from a distribution (latin hypercube sampling).
    # (data = distribution, nsample = number of desired samples)

    def lhs(self,data):
        P = 100*(np.random.permutation(self.n_samples) + 1 - np.random.uniform(size=(self.n_samples)))/self.n_samples
        s = np.percentile(data, P)
        return s

    # Function generates distribution range for each parameter and performs latin hypercube sampling.
    def parameter_sampler(self,size=1000000):

        samples = {}
        for p in self.params:
            if len(self.params[p]) == 2: # If 2 numbers in parameter range we do sampling.

                distribution = self.loguniform(size=size)
                # Perform lhs using the resulting distribution.
                samples[p] = self.lhs(distribution)
            elif len(self.params[p]) == 1: # If one number in parameter range it is constant.
                samples[p] = self.params[p][0]

        return pd.DataFrame(samples)

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
        centraldiag = [1.+alphan]+[1.+2.*alphan for j in range(J-2)]+[1.+alphan]
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
        # rewrite to include hill coefficients
        if interaction == 0:
            # No interaction between morphogens
            return lambda concentration: 1

        if interaction == 1:
            # Activation equation
            return lambda concentration: 1/(1 + (rate/concentration)) # **hill_coefficient

        if interaction == -1:
            # Repression equation
            return lambda concentration: 1/(1 + (concentration/rate)) # **hill_coefficient

    def odes(concentration, production_x, degradation_x, max_conc_x, production_y, degradation_y, max_conc_y, hill_equations, **params):
        # Calculate system of ODEs for concentration grid
        # Currently written for two morphogens
        f_A = production_x + max_conc_x * (hill_equations["xx"](concentration[0]) * hill_equations["yx"](concentration[1]) - degradation_x)
        f_B = production_y + max_conc_y * (hill_equations["xy"](concentration[0]) * hill_equations["yy"](concentration[1]) - degradation_y)

        return np.array([f_A, f_B])

    def solve(self, a_matrix, b_matrix, concentration, ode):
        # Solve grid concentrations for one time step
        return np.dot(a_matrix, (np.dot(b_matrix, concentration) + ode))

def create_grids(adjmatrix, dimensions):
    # Create concentration grids for one adjacency matrix
    conc_grids=[]
    for i in range(adjmatrix.shape[0]):
        conc_grids.append(Grid.create(dimensions=dimensions))

    return conc_grids

def plot_conc(U):
    plt.plot(U[0], label='U')
    plt.plot(U[1], label='V')
    plt.xlabel('Space')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()

def parse_args(inputs):
    # Function for parsing command line arguments
    args = dict(
        num_nodes=2,
        dimensions=(300,),
        system_length=300,
        total_time=299,
        num_samples=100,
        growth=None
    )

    for a in inputs:

        if "-" in a:
            try:
                command_line_input = int(inputs[inputs.index(a) + 1])
            except:
                command_line_input = inputs[inputs.index(a) + 1]
            args[a[1:]] = command_line_input

    return args

if __name__ == '__main__':

    args = parse_args(sys.argv)

    adjmatrices = Topologies().create_adjacency_matrices(num_nodes=args["num_nodes"])
    sampler = LHS()
    parameters = sampler.parameter_sampler()

    J = args["system_length"]
    dx = float(J)/(float(J)-1)
    x_grid = np.array([j*dx for j in range(J)])

    total_time = args["total_time"]
    num_timepoints = 10*total_time
    dt = float(total_time) / float(num_timepoints - 1)
    alpha_solve = lambda x: Solver.calculate_alpha(x, dt, dx)


    print("Finding alphan values...")
    for param in tqdm(parameters.columns):
        if "diffusion" in param:
            parameters[f"alphan_{param[-1]}"] = parameters[param].apply(alpha_solve)

    concentration_grids = []
    for adj in adjmatrices:
        concentration_grids.append(create_grids(adj,args["dimensions"]))

    if args["num_nodes"] == 2:
        start = datetime.now()
        print("Starting sample testing...")
        results = dict()
        for row in tqdm(range(args["num_samples"])):
            parameter_set = parameters.loc[row].to_dict()
            if args["growth"] == None:
                A_matrices = [[Solver.a_matrix(parameter_set["alphan_x"],J),Solver.a_matrix(parameter_set["alphan_y"],J)]]
                B_matrices = [[Solver.b_matrix(parameter_set["alphan_x"],J),Solver.b_matrix(parameter_set["alphan_y"],J)]]

            if args["growth"] == "linear":
                A_matrices = [[Solver.a_matrix(parameter_set["alphan_x"],j+1),Solver.a_matrix(parameter_set["alphan_y"],j+1)] for j in range(J)]
                B_matrices = [[Solver.b_matrix(parameter_set["alphan_x"],j+1),Solver.b_matrix(parameter_set["alphan_y"],j+1)] for j in range(J)]


            for adj, concentrations in zip(adjmatrices, concentration_grids):
                parameter_set_copy = copy.deepcopy(parameter_set)
                parameter_set_copy["hill_equations"] = dict(
                    xx=Solver.hill_equations(adj[0,0], parameter_set_copy["k_xx"]),
                    yx=Solver.hill_equations(adj[0,1], parameter_set_copy["k_yx"]),
                    xy=Solver.hill_equations(adj[1,0], parameter_set_copy["k_xy"]),
                    yy=Solver.hill_equations(adj[0,1], parameter_set_copy["k_yy"])
                    )

                A_matrix = A_matrices[0]
                B_matrix = B_matrices[0]

                for ti in range(num_timepoints):
                    concentrations_new = copy.deepcopy(concentrations)

                    f0 = Solver.odes(concentration=concentrations,**parameter_set_copy)*(dt/2)
                    for n in range(2):
                        concentrations_new[n] = np.dot(A_matrix[n], (B_matrix[n].dot(concentrations_new[n]) +  f0[n]))

                    hour = ti / (num_timepoints / total_time)

                    if args["growth"] == "linear" and hour % 1==0:
                        concentrations_new = [Grid.growth(c) for c in concentrations_new]
                        A_matrix = A_matrices[int(hour)+1]
                        B_matrix = B_matrices[int(hour)+1]

                    concentrations = copy.deepcopy(concentrations_new)
                #plot_conc(concentrations)
                #input()
                #results[][str(adj)] = concentrations

    print(datetime.now()-start)
    print(pd.DataFrame(results))
    # insert main script running here
