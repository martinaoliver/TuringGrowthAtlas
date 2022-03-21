import pickle
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import tqdm

def FilterNone(results):
    binme = {}
    for i in results:
        if results[i]['concs'] == None:
            binme[i] = results[i]
    for i in binme:
        results.pop(i)
    return(results)    

with open('2022-03-10_14-19Linear_Results.pkl', 'rb') as f:
    growth = pickle.load(f)

growth = FilterNone(growth)    
    
with open('2022-03-10_14-19None_Results.pkl', 'rb') as f:
    nongrowth = pickle.load(f)
    
nongrowth = FilterNone(nongrowth)    
    
    
def fourier_classify(U, threshold=2, growth = False, plot=False):
      
    U = [[round(conc, 3) for conc in morphogen] for morphogen in U]
    
    if growth:
        U = [conc[20:80] for conc in U]
    
    # Compute the fourier transforms.
    transforms = [fft(i) for i in U]

    # Check for peaks.
    x_fpeaks = [abs(i) > threshold for i in transforms[0][1:]]
    y_fpeaks = [abs(i) > threshold for i in transforms[1][1:]]

    # Peaks must be found in both species to return True.
    fpeaks_found = False
    if sum(x_fpeaks) > 0 and sum(y_fpeaks) > 0:
        fpeaks_found = True
    
    peaks = [len(find_peaks(i)[0]) for i in U]
    multiple_peaks_found = False
    for i in peaks:
        if i > 2:
            multiple_peaks_found = True
    
    return fpeaks_found and multiple_peaks_found

print('Checking growth...')
    
growth_hits = {}
for i in tqdm(growth):
    if fourier_classify(growth[i]['concs'], growth = True):
        growth_hits[i] = growth[i]
        
print('Growth hits: ', len(growth_hits))   
print('Checking non-growth...')

nongrowth_hits = {}
for i in tqdm(nongrowth):
    if fourier_classify(nongrowth[i]['concs'], growth = False):
        nongrowth_hits[i] = nongrowth[i]

print('Non-growth hits: ', len(nongrowth_hits))



      