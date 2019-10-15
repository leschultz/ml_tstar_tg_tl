import pickle
import os
import re

export = 'systems.txt'  # Pickle file containing available systems

# Potential locations
ben = 'ben_potentials'
eam = 'eam_potentials.txt'
nist = 'nist_potentials.txt'

# Potentials from Ben
potentials = os.listdir(ben)
potentials = [i.split('.')[0] for i in potentials]
potentials = [re.findall('[A-Z][a-z]?|[0-9]+', i) for i in potentials]
potentials = [set(i) for i in potentials]

# Potentials from NIST
with open(nist) as f:
    for line in f:
        value = set(line.strip().split('-'))
        potentials.append(value)

# Potentials from EAM Dr. sheng
with open(eam) as f:
    for line in f:
        values = line.strip().split(' ')
        values = [i for i in values if i != '']
        values = [i for i in values if ':' not in i]
        values = [set(i.split('-')) for i in values]

        if not values:
            continue

        potentials += values

with open(export, 'wb') as f:
    pickle.dump(potentials, f)
