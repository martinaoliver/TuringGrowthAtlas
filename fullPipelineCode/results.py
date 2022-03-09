import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.Spectral
from solver import Solver
import numpy as np
from matplotlib.animation import FuncAnimation

def plot_conc(U):

    fig, ax1 = plt.subplots()
    color = 'tab:green'
    ax1.set_xlabel('Space')
    ax1.set_ylabel('Concentration x', color=color)
    ax1.plot(U[0], color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Concentration y', color=color)
    ax2.plot(U[1], color=color)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.show()

def surfpattern(results, morphogen=0):
    x_grid = np.array([j*0.6 for j in range(50)])
    t_grid = np.array([j*(1500/17999) for j in range(len(results))])
    results = np.vstack(results)
    results = np.transpose(results)
    r_non_zero = results[results != 0]
    levels = [0.]
    # for i in range(10):
    #     levels.append(np.percentile(r_non_zero, 10 * i, interpolation='midpoint'))
    # levels = list(dict.fromkeys(levels))
    t, x = np.meshgrid(t_grid, x_grid)
    plt.contourf(x, t, results, cmap=cmap)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

conc = results[0]

def set_fig(conc):
    stacked = np.vstack(conc)
    fig = plt.figure()
    axis = plt.axes(xlim=(0,49),
                    ylim=(np.max(stacked),np.min(stacked)
                    )
    line, = axis.plot([], [], lw = 3)
    return fig, axis, line

def init():
    line.set_data([], [])
    return line

def animate(i):
    x = np.array([i for i in range(50)])
    y = conc[i]
    line.set_data(x,y)
    return line

anim = FuncAnimation(fig, animate, init_function=init)


g = "/Users/sammoney-kyrle/Downloads/0303_Growth.pkl"
n = "/Users/sammoney-kyrle/Downloads/0303_NonGrowth.pkl"
p = "/Users/sammoney-kyrle/Downloads/parameters2.pkl"

with open(g, "rb") as f:
    g = pickle.load(f)
with open(n, "rb") as f:
    n = pickle.load(f)
with open(p, "rb") as f:
    p = pickle.load(f)

pattern_param = dict()
all_r = list()

count=0
for k in g:

    params = p[(k[0],k[1])]
    g[k]["params"] = params
    try:
        if g[k]["Fourier"][0] and g[k]["Fourier"][1]:
            pattern_param[(k[0],k[1])] = params
                # pattern_param[(k[0],k[1])] = params
        elif g[k]["LSA"]["system_class"] != "simple unstable" and g[k]["LSA"]["system_class"] != "simple stable" and g[k]["LSA"]["system_class"] != "unclassified":
            pattern_param[(k[0],k[1])] = params
            # count = count+1

            # conc_list, SteadyState_list, LSA_list, fourier_list = Solver.solve(params[0],params[1],"linear",1500./17999.,0.6,50,1500,18000,save_all=True)
            # surfpattern(conc_list[0])
            # input()
    except:
        pass
# with open("/Users/sammoney-kyrle/Downloads/growth_parameters2.pkl","wb") as f:
#     pickle.dump(pattern_param,f)


# all_r = list()
count=0
for k in n:
    params = p[(k[0],k[1])]
    n[k]["params"] = params
    try:
        if n[k]["Fourier"][0] and n[k]["Fourier"][1]:
            pattern_param[(k[0],k[1])] = params
            count = count+1
        elif n[k]["LSA"]["system_class"] != "simple unstable" and n[k]["LSA"]["system_class"] != "simple stable" and n[k]["LSA"]["system_class"] != "unclassified":
            pattern_param[(k[0],k[1])] = params

    except:
        pass
# with open("/Users/sammoney-kyrle/Downloads/nongrowth_parameters2.pkl","wb") as f:
#     pickle.dump(pattern_param,f)
count=0

print(len(pattern_param.keys()))
with open("/Users/sammoney-kyrle/Downloads/all_parameters_2.pkl","wb") as f:
    pickle.dump(pattern_param,f)
