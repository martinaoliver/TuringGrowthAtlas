from pathlib import Path
import pickle
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def FilterNone(results):
    binme = {}
    for i in results:
        if results[i]['concs'] == None:
            binme[i] = results[i]
    for i in binme:
        results.pop(i)
    return(results)

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

growth_robustness = {}
nongrowth_robustness = {}

pathlist = Path('neighbourhood').rglob('*.pkl')
for path in pathlist:

     path_in_str = str(path)
     #print(path_in_str)
    
     bracket = path_in_str.index(')')
     
     if 'results' in path_in_str:
         
         infile = open(path_in_str, 'rb')
         results = pickle.load(infile)
         infile.close()
         results = FilterNone(results)
         
         growth = True
         if 'Growth' in path_in_str:
            hit_counter = 0    
            for i in results:
                if fourier_classify(results[i]['concs'], growth = growth):
                    hit_counter += 1
                 
            growth_robustness[path_in_str[14:(bracket+1)]] = (hit_counter, len(results))        
#         if growth:
 #            growth_robustness[path_in_str[14:(bracket+1)]] = (hit_counter, len(results))
  #       elif not growth:
   #          nongrowth_robustness[path_in_str[14:(bracket+1)]] = (hit_counter, len(results))
                
     elif 'parameters' in path_in_str:
         pass
     
with open('growth_robustness3.pkl', 'wb') as f:
    pickle.dump(growth_robustness, f)
    
#with open('nongrowth_robustness2.pkl', 'wb') as f:
#    pickle.dump(nongrowth_robustness, f)
