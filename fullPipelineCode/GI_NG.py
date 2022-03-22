import pickle

gi = [(1471, 15, 0),
     (1471, 16, 0),
     (2045, 15, 0),
     (5177, 4, 0),
     (5177, 16, 0),
     (6218, 15, 0),
     (6218, 16, 0),
     (13964, 17, 0),
     (31782, 16, 0),
     (38381, 18, 0),
     (38381, 20, 0),
     (40812, 15, 0),
     (46221, 18, 0)]

print('Loading data...')

with open('name.pkl', 'rb') as f:
    data = pickle.load(f)
    
ng_gi = {}

for i in gi:
    ng_gi[i] = data[i]
    
with open('Growth-induced_NonGrowth.pkl', 'wb') as f:
    pickle.dump(ng_gi, f)