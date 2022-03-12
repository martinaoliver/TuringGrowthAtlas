import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


##################### IMPORT DATA ############################

# These files are pre-filtered hits from the 10/03 analysis.

infile = open('1003_LSA_hits.pkl', 'rb')
LSA = pickle.load(infile)
infile.close()

infile = open('1003_growth_fourier.pkl', 'rb')
growth = pickle.load(infile)
infile.close()

infile = open('1003_nongrowth_fourier.pkl', 'rb')
nongrowth = pickle.load(infile)
infile.close()

# Parameter set 42.

infile = open('1003_Results_ParamSet42/parameters.pkl', 'rb')
params = pickle.load(infile)
infile.close()


########################## PCA ############################

# Remove noise.
# LSA = rmNoise(LSA)

# Make df with parameters as headings.
category = []
tp_params = []

for i in LSA:
    
    # Add tp_class.
    category.append(LSA[i]['LSA']['system_class'])
    
    # Add parameters to list of dictionaries.
    tp_params.append(params[(i[0], 0)][0])
    
df = pd.DataFrame(tp_params)

# Scale the data.
scaler = StandardScaler()
scaler.fit(df)
df = scaler.fit_transform(df)

# PCA with 2 components.
pca = PCA(n_components = 2)
df_new = pca.fit_transform(df)
principalDf = pd.DataFrame(data = df_new, columns = ['PC1', 'PC2'])
finalDf = pd.concat([principalDf, pd.DataFrame(category, columns = ['Turing_Class'])], axis = 1)

# Show how effective the PCA is at explaining variance.
print('Explained variance: ', pca.explained_variance_ratio_)

# Plot the PCA outcome.
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['turing I', 'turing II', 'turing I oscillatory', 'hopf', 'turing I hopf', 'turing II hopf']
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Turing_Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c = color, s = 50)
ax.legend(targets)
ax.grid()
