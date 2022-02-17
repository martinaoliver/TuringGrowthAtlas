# Author Xindong
# Date 2022/2/6 23:19
import numpy as np
from scipy import optimize
from sympy import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def noncompetitiveact(U, km,n=2):
    act = ((U / km) ** (n)) / (1 + (U / km) ** (n))
    return act

def noncompetitiveinh(U, km,n=2):
    inh = 1 / (1 + (U / km) ** (n))
    return inh

def turing_hill(U):
    A,B=U
    ba,Va,kaa,kba,muA =  0.1,100,10,10,1
    bb,Vb,kab,muB = 0.1,100,100,40
    fA = ba + Va * noncompetitiveact(A,kaa) * noncompetitiveinh(B,kba) - muA*A
    fB = bb + Vb * noncompetitiveact(A,kab) - muB*B
    return (fA,fB)

def jacobian1():
    A,B = symbols('A'),symbols('B')
    arguments = Matrix([A,B])
    functions = Matrix(turing_hill([A,B]))
    jacobian_topology = functions.jacobian(arguments)

    return jacobian_topology


#provide the initial values
#note that we want as much initial values as possible to reach any possible steady state
def loguniform(low=-3, high=3, size=None):
    return (10)**(np.random.uniform(low, high, size))

def lhs_list(data,nsample):
    nvar = data.shape[1]
    ran = np.random.uniform(size=(nsample,nvar))
    s = np.zeros((nsample,nvar))
    for j in range(0,nvar):
        idx = np.random.permutation(nsample)+1
        P = ((idx-ran[:,j])/nsample)*100
        s[:,j] = np.percentile(data[:,j],P)
    return s

def lhs_initial_conditions(n_initialconditions=10,n_species=2):
    data = np.column_stack(([loguniform(size=100000)]*n_species))
    initial_conditions = lhs_list(data,n_initialconditions)
    return np.array(initial_conditions,dtype=np.float)


#define newton raphson approach
def newton_raphson(x_initial, max_num_iter=15, tolerance=0.0001, alpha=1):

    #initialize
    x = x_initial
    fx = turing_hill(x)
    err = np.linalg.norm(fx) #Euclid norm, norm[x,y] = root_square(x^2 + y^2)
    iter = 0

    # perform the Newton-Raphson iteration
    while err > tolerance and iter < max_num_iter and np.all(x != 0):
        jac = jacobian1()
        A, B = symbols('A'), symbols('B')
        jac = jac.subs(A, x[0])
        jac = jac.subs(B, x[1])
        jac = np.array(jac, dtype=float)

        #Newton step
        x = x - alpha * np.linalg.solve(jac, fx)
        fx = turing_hill(x)
        err = np.linalg.norm(fx)
        iter = iter + 1

    # check that there are no negatives
    if err < tolerance:
        if sum(item < 0 for item in x) == 0:
            return (x, err, 0)


def newtonraphson_run(initial_conditions):
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
                        np.allclose(SteadyState_list[i], xn[0], rtol=10 ** -2, atol=0))  # PROCEED IF NO TRUES FOUND
                if not True in logiclist:  # no similar steady states previously found
                    SteadyState_list.append(xn[0])
                    count += 1

    return SteadyState_list

if __name__=='__main__':

    np.random.seed(2)
    initial_conditions1 = lhs_initial_conditions(n_initialconditions=100,
                                                 n_species=2)
    SteadyState_list = newtonraphson_run(initial_conditions1)

    print("Number of steady state:" + str(len(SteadyState_list)))
    for i in SteadyState_list:
        print(i)

