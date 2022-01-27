import numpy as np
import itertools
from tqdm import tqdm
import pickle

N = 3 #number of nodes

# Make array with all possible ajdacency matrices with an activation (1), inhibition (-1) or no connection (0).
initial = [np.reshape(np.array(i), (N,N)) for i in itertools.product([0, 1,-1], repeat = N*N)] 
adj_matrix_array = initial.copy()

def is_connected(M): #This function checks if a network M is connected.

    M1 = M.copy() # Create copy so initial array does not get modified
    np.fill_diagonal(M1, 0) # Connections to itself do not count when identifying if a network is unconnected. Therefore we set those to zero (diagonal).

    # Every column and row must have at least one non-zero value for the network to be connected.
    if all(M1.any(axis=0)):
        if all(M1.any(axis=1)):
            return True

def remove_unconnected(matrix_array): # This function loops through the array of networks and deletes the UNCONNECTED networks.
    
    new_matrix_array = []
    
    for index,M in enumerate(matrix_array): # Loop through matrices.
        if is_connected(M) == True: # If matrix connected, add to new array.
            new_matrix_array.append(M)
            
    return new_matrix_array

def isomorphs(A): # This function returns sorted list of isomorphisms of graph AH.

    # (see https://www.youtube.com/watch?v=UCle3Smvh1s&t=409s for understanding).
    
    # There are six permutation matrices for the 3x3 case.
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

def remove_isomorphism(M): # This function loops through matrices removing isomorphisms.
    
    M_dict = {} # Dictionary to be populated with unique matrices.
    
    with tqdm(total = len(M)) as pbar:
        
        # For each netwoek, get permutations of that network.
        # Check if that set of permutations is already a key and if not, add them.
    
        for network in M:
            
            permutations = isomorphs(network)
            
            if permutations not in M_dict:
                
                M_dict[permutations] = network
                
            pbar.update(1)
    
    return M_dict.values()

adj_matrix_array_connected = remove_unconnected(adj_matrix_array)

atlas_matrix_array = remove_isomorphism(adj_matrix_array_connected)

atlas_matrix_array = list(atlas_matrix_array)

print(len(atlas_matrix_array))

with open('Matrices_3N3D.pkl','wb') as f:
    pickle.dump(atlas_matrix_array, f)