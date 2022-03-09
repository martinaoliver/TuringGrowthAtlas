import numpy as np
import itertools

class Atlas:

    def __init__(self, nodes = 2, diffusers = 2):
        self.nodes = nodes
        self.diffusers = diffusers

    def is_connected(self, M): #This function checks if a network M is connected.

        M1 = M.copy() # Create copy so initial array does not get modified
        np.fill_diagonal(M1, 0) # Connections to itself do not count when identifying if a network is unconnected. Therefore we set those to zero (diagonal).

        # Every column and row must have at least one non-zero value for the network to be connected.
        if all(M1.any(axis=0)):
            if all(M1.any(axis=1)):
                return True

    def remove_unconnected(self, matrix_array): # This function loops through the array of networks and deletes the UNCONNECTED networks.

        new_matrix_array = []

        for index,M in enumerate(matrix_array): # Loop through matrices.
            if self.is_connected(M) == True: # If matrix connected, add to new array.
                new_matrix_array.append(M)

        return new_matrix_array

    def isomorphs(self, A, nodes, diffusers): # This function returns sorted list of isomorphisms of graph A.

        # (see https://www.youtube.com/watch?v=UCle3Smvh1s&t=409s for understanding).

        if nodes == 2 and diffusers == 2:
            P_array = [np.matrix('0 1; 1 0'), np.matrix('1 0; 0 1')]

        elif nodes == 3 and diffusers == 2:
            P_array = [np.matrix('1 0 0; 0 1 0; 0 0 1'), np.matrix('0 1 0; 1 0 0; 0 0 1')]

        elif nodes == 3 and diffusers == 3:
            P_array = [np.matrix('1 0 0; 0 0 1; 0 1 0'), np.matrix('0 1 0; 1 0 0; 0 0 1'),\
                       np.matrix('0 1 0; 0 0 1; 1 0 0'), np.matrix('0 0 1; 0 1 0; 1 0 0'),\
                       np.matrix('0 0 1; 1 0 0; 0 1 0'), np.matrix('1 0 0; 0 1 0; 0 0 1')]

        # Get transpose matrices for each.
        PT_array = [np.transpose(P) for P in P_array]
        P_arrays = zip(P_array, PT_array)

        # Get all the permutations of matrix A and sort them.
        permutations = [np.dot(np.dot(P,A),PT) for (P,PT) in P_arrays]
        permutations = list(map(str,permutations))
        permutations.sort()
        permutations = ''.join(permutations)
        return permutations

    def remove_isomorphism(self, M, nodes, diffusers): # This function loops through matrices removing isomorphisms.

        M_dict = {} # Dictionary to be populated with unique matrices.

            # For each netwoek, get permutations of that network.
            # Check if that set of permutations is already a key and if not, add them.

        for network in M:

            permutations = self.isomorphs(network, nodes, diffusers)

            if permutations not in M_dict:

                M_dict[permutations] = network

        return M_dict.values()

    def create_adjacency_matrices(self, nodes, diffusers):

        # Make array with all possible adjacency matrices with an activation (1), inhibition (-1) or no connection (0).
        initial = [np.reshape(np.array(i), (nodes, nodes)) for i in itertools.product([0, 1,-1], repeat = nodes*nodes)]
        adj_matrix_array = initial.copy()
        adj_matrix_array_connected = self.remove_unconnected(adj_matrix_array)
        #atlas_matrix_array = self.remove_isomorphism(adj_matrix_array_connected, nodes = nodes, diffusers = diffusers)
        atlas_matrix_array = {count: array for count, array in enumerate(adj_matrix_array_connected)}
        return atlas_matrix_array
