import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.Spectral

infile = open('linear(5579, 0)_results.pkl', 'rb')
growth = pickle.load(infile)
infile.close()

hits = {}
for i in growth:
    if sum(growth[i]['Fourier']) == 2:
        hits[i] = growth[i]
        
print(len(hits))
        
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
    
for i in hits:
    plot_conc(hits[i]['concs'])