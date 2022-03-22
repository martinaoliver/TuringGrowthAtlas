import pickle

gi = [(1471, 15),
     (1471, 16),
     (2045, 15),
     (5177, 4),
     (5177, 16),
     (6218, 15),
     (6218, 16),
     (13964, 17),
     (31782, 16),
     (38381, 18),
     (38381, 20),
     (40812, 15),
     (46221, 18)]

print('Loading data...')

with open('2022-03-10_14-19None_results.pkl', 'rb') as f:
    data = pickle.load(f)
      
print('Checking for growth-induced patterns...')    
    
ng_gi = {}

for i in data:
    if (i[0], i[1]) in gi:
        ng_gi[i] = data[i]
    
print('Results: ', ng_gi)

print('Saving...')

with open('Growth-induced_NonGrowth.pkl', 'wb') as f:
    pickle.dump(ng_gi, f)
