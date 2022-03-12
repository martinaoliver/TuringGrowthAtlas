# libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import pickle
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import numpy as np
from sklearn.preprocessing import StandardScaler



###################### LOAD DATA AND PARAMETERS ####################

infile = open('1003_LSA_hits.pkl', 'rb')
LSA = pickle.load(infile)
infile.close()

infile = open('1003_growth_fourier.pkl', 'rb')
growth = pickle.load(infile)
infile.close()

infile = open('1003_nongrowth_fourier.pkl', 'rb')
nongrowth = pickle.load(infile)
infile.close()

infile = open('1003_Results_ParamSet42/parameters.pkl', 'rb')
params = pickle.load(infile)
infile.close()



############# FUNCTIONS TO REMOVE NOISE FROM DATA #################

def fourier_classify(U, threshold=2, plot=False, title = 'plot'):

    # Round to 5sf avoid picking up noise.
    U = [np.array([np.format_float_positional(conc, precision=5, unique=False, fractional=False, trim='k') for conc in morphogen]) for morphogen in U]

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
        for i in [0, 1]:
            
            amplitudes = transforms[i]
            freq = fftfreq(100, 0.2)
            plt.plot(freq[1:], abs(amplitudes)[1:])
            plt.xlim(0, )
            plt.ylim(1,)
            plt.title(f'Prominent frequency: {freq[np.argmax(abs(amplitudes)[1:])]}')
            plt.show()

    return peaks_found

def rmNoise(results):
    
    filtered = {}
    for i in results:
        if fourier_classify(results[i]['concs']):
            filtered[i] = results[i]
            
    return(filtered)

LSA = rmNoise(LSA)



############################ MAKE PLOT ############################

def ParallelPlot(results, params):

    category = []
    tp_params = []
    
    for i in results:
        
        # Add tp_class.
        category.append(LSA[i]['LSA']['system_class'])
        
        # Add parameters to list of dictionaries.
        tp_params.append(params[(i[0], 0)][0])
    
    # Convert dict list to dataframe of params.
    df = pd.DataFrame(tp_params)
    
    # Scale parameter values.
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns = list(tp_params[0].keys()))
    
    # Append categories to dataframe.
    finalDf = pd.concat([df, pd.DataFrame(category, columns = ['Turing_Class'])], axis = 1)
    
    # Make the plot
    parallel_coordinates(finalDf, 'Turing_Class', colormap=plt.get_cmap("Set2"))
    plt.xticks(rotation=90)
    plt.title('Parameter Values')
    plt.ylabel('Scaled parameter values')
    plt.xlabel('Parameter')
    plt.show()
    
    
############################ RUN ############################

LSA = rmNoise(LSA)
ParallelPlot(LSA, params)