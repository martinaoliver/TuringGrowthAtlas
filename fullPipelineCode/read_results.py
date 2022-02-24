### Read results.

import pickle

infile = open('1t_results.pkl','rb')
results_dict = pickle.load(infile)
infile.close()

for i in results_dict:
    if results_dict[i]['Fourier'] == True:
        print(results_dict[i])