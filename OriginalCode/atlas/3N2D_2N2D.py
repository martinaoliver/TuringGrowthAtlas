import numpy as np
import itertools
from tqdm import tqdm
import pickle

N = 3 # Number of nodes

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

def isomorphic(AH): #This function returns list of isomorisms of graph AH.

    if N==2:
        
        P = np.matrix('0 1; 1 0')
        PT = np.transpose(P)
        permutation = np.array_str(np.dot(np.dot(P,AH),PT))
        return permutation
    
    if N==3:
        
        P = np.matrix('0 1 0; 1 0 0; 0 0 1')
        PT = np.transpose(P)
        permutation = np.array_str(np.dot(np.dot(P,AH),PT))
        return permutation


def remove_isomorphism(M):
    to_pop = []
    with tqdm(total=len(M.keys())) as pbar:
        for key in M:

            if key in to_pop:
                continue

            elif key == M[key]:
                continue

            elif M[key] in M:
                to_pop.append(M[key])

            else:
                continue
            pbar.update(1)

    for r in to_pop:
        M.pop(r)

    return M


adj_matrix_array_connected = remove_unconnected(adj_matrix_array)
adj_matrix_array_connected = {np.array_str(A): isomorphic(A) for A in adj_matrix_array_connected}
atlas_matrix_array = remove_isomorphism(adj_matrix_array_connected)
atlas_matrix_array = [np.array(key) for key in atlas_matrix_array.keys()]
print(len(atlas_matrix_array))

with open('Matrices_3N2D.pkl','wb') as f:
    pickle.dump(atlas_matrix_array, f)