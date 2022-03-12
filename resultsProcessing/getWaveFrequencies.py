import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.Spectral
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import csv
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq



############# IMPORT DATA ####################

infile = open('1003_LSA_hits.pkl', 'rb')
LSA = pickle.load(infile)
infile.close()

infile = open('1003_growth_fourier.pkl', 'rb')
growth = pickle.load(infile)
infile.close()

infile = open('1003_nongrowth_fourier.pkl', 'rb')
nongrowth = pickle.load(infile)
infile.close()




################ FILTER NOISE #####################

def rmNoise(results):
    
    filtered = {}
    for i in results:
        if fourier_classify(results[i]['concs']):
            filtered[i] = results[i]
            
    return(filtered)



########### MODIFIED FOURIER ##################

def fourier_classify(U, threshold=2, plot=False, title = 'plot', label = '', growth = False, return_freq = False):

    # Round to 5sf avoid picking up noise.
    U = [np.array([np.format_float_positional(conc, precision=5, unique=False, fractional=False, trim='k') for conc in morphogen]) for morphogen in U]

    if growth:
        U = [conc[20:80] for conc in U]
        
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
        if peaks_found:
            for i in [0, 1]:         
                amplitudes = transforms[i]
                freq = fftfreq(100, 0.2)
                plt.plot(freq[1:], abs(amplitudes)[1:])
                plt.xlim(0, )
                plt.ylim(1,)
                plt.title(f'{label} Prominent frequency: {freq[np.argmax(abs(amplitudes)[1:])]}')
                plt.show()
        else:
            print(label, ': no peaks')
          
    # Return wave frequency.        
    if return_freq:
        freqs = []
        for i in [0,1]:               
            amplitudes = transforms[i]
            freq = fftfreq(100, 0.2)
            freqs.append(round(freq[np.argmax(abs(amplitudes)[1:])], 2))
        return freqs
            
    return peaks_found




def plot_conc(U, title = 'Plot'):

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

    plt.title(title)
    fig.tight_layout()
    plt.show()


############# GET WAVE FREQUENCIES ################

LSA = rmNoise(LSA)

frequencies = {}
for i in LSA:
    frequencies[i] = fourier_classify(LSA[i]['concs'], return_freq = True)
#     plot_conc(LSA[i]['concs'], title = i)
      
    
