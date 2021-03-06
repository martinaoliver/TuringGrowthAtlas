import numpy as np
import itertools


##################################
#STEP1: CREATE ARRAY OF ADJACENCY MATRICES
N = 2 #number of nodes (can only work with 2 node networks for now)
initial = [np.reshape(np.array(i), (N,N)) for i in itertools.product([0, 1,-1], repeat = N*N)] #array with all possible ajdacency matrices with an activation (1), inhibition (-1) or no connection (0).
adj_matrix_array = initial.copy()
#######################################

#################################
#STEP2 functions:REMOVE UNCONNECTED NETWORKS
#Definition of CONNECTED NETWORK: A network with an incoming and outgoing edge.
def is_connected(M): #This function checks if a network M is connected (If so, it returns True)
    M1= M.copy() #create copy so initial array does not get modified
    np.fill_diagonal(M1, 0) #connections to itself do not count when identifying if a network is unconnected. Therefore we set those to zero (diagonal).

    ##Every column and row must have at least one non-zero value for the network to be connected.
    if all(M1.any(axis=0)):
        if all(M1.any(axis=1)):
            return True


def remove_unconnected(matrix_array): #This function loops through the array of networks and deletes the UNCONNECTED networks.
    new_matrix_array = []
    for index,M in enumerate(matrix_array): #loop through matrices
        if is_connected(M)==True: #if matrix connected, add to new array
            new_matrix_array.append(M)
    return new_matrix_array
#################################

#################################
#STEP3 functions: REMOVE ISOMORPHISMS
#Definition of NETWORK ISOMORPHISM: Two graphs related by isomorphism differ only by the names of the vertices and edges. There is a complete structural equivalence between two such graphs.
#For the mathematical theory behind checking isomorphisms using adjacency matrices go to https://iuuk.mff.cuni.cz/~andrew/DMex11s.pdf or https://www.youtube.com/watch?v=UCle3Smvh1s.
def is_isomorphic(AG,AH): #This function checks if two graphs AG and AH are isomorphic.
    if N==2:
        P = np.matrix('0 1; 1 0')
    #Code only works for 2 node networks for now. Need to figure out which permutation matrices to use for three node networks.
    # if N==3:
    #    P1 = np.matrix('0 1 0; 1 0 0; 0 0 0');  P1 = np.matrix('0 0 0; 0 0 1; 0 1 0');  P1 = np.matrix('0 0 1; 0 0 0; 1 0 0');  P1 = np.matrix('0 1 0; 0 0 1; 1 0 0');

    PT = np.transpose(P)
    if np.all(AG == np.dot(np.dot(P,AH),PT)): #if this statement is true, the graphs AH and AG are isomorphic
        return True
    else:
        return False

def remove_isomorphism(matrix_array): #this function checks if the ISOMORPHIC matrices exist in the array and removes them to avoid symmetric matrices
#careful as symmetric matrices around diagonal are isomorphic to itself. can only check against other matrices
    for index,AG in enumerate(matrix_array):

        if any(is_isomorphic(AG,AH) for AH in matrix_array): #if AG has an isomorph in the array of graphs:
            i = int(np.where([is_isomorphic(AG,AH) for AH in matrix_array])[0]) #Check where this isomorph AH is
            if i != index: #if AG=!AH (to check that AG is not being classified as isomorphic to itself (AG=AH))
                matrix_array.pop(i) #in that case, remove AH from array of matrices
    return matrix_array
#################################

#Execute functions for removing unconnected and isomorphic networks.
adj_matrix_array_connected = remove_unconnected (adj_matrix_array)
atlas_matrix_array = remove_isomorphism(adj_matrix_array_connected)

print(np.shape(atlas_matrix_array))
for i,M in enumerate(atlas_matrix_array):
    print(i)
    print(M)
    print('')
