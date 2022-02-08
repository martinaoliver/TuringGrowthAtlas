import numpy as np
from scipy import optimize
from sympy import *
from Part2_PARAMETERS import LHS

class Steady_State:

    def __init__(self,react_func):
        self.react_func = [] #assign the react function in the solver

    def update_react(self, react_ss):
        self.react_func = react_ss

    def jacobian1(self):
        X, Y = symbols('X'), symbols('Y')
        arguments = Matrix([X, Y])
        functions = Matrix(react_func([X, Y]))
        jacobian_topology = functions.jacobian(arguments)
        return jacobian_topology

    def lhs_initial_conditions(self, n_initialconditions=10, n_species=2):
        data = np.column_stack(([LHS.loguniform(size=100000)] * n_species))
        initial_conditions = LHS.lhs(data, n_initialconditions)
        return np.array(initial_conditions, dtype=np.float)

    def newton_raphson(self, x_initial, max_num_iter=15, tolerance=0.0001, alpha=1):
        x = x_initial
        fx = react_func(x)
        err = np.linalg.norm(fx)
        iter = 0

        # perform the Newton-Raphson iteration
        while err > tolerance and iter < max_num_iter and np.all(x != 0):
            jac = jacobian1()
            X, Y = symbols('X'), symbols('Y')
            jac = jac.subs(X, x[0])
            jac = jac.subs(Y, x[1])
            jac = np.array(jac, dtype=float)

            # update
            x = x - alpha * np.linalg.solve(jac, fx)
            fx = react_func(x)
            err = np.linalg.norm(fx)
            iter = iter + 1

        # check that there are no negatives
        if err < tolerance:
            if sum(item < 0 for item in x) == 0:
                return (x, err, 0)

    def newtonraphson_run(self, initial_conditions):
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
