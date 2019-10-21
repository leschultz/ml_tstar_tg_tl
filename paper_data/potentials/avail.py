import pickle
import os
import re

export = 'systems.txt'  # Pickle file containing available systems

# Potential locations
ben = 'ben_potentials'
eam = 'eam_potentials.txt'
nist = 'nist_potentials.txt'
kim = 'openkim_potentials.txt'

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

# Potentials from OpenKim
with open(kim) as f:
    for line in f:
        line = line.strip()
        line = line.split('_')
        line = line[-5:]

        if not line:
            continue

        line = line[0]
        line = set(re.findall('[A-Z][a-z]?|[0-9]+', line))

        if not line:
            continue

        potentials.append(line)

with open(export, 'wb') as f:
    pickle.dump(potentials, f)
