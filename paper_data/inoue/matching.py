from matplotlib import pyplot as pl

import pandas as pd
import numpy as np

import pickle
import json
import os
import re

inoue = 'data.txt'
potentials = '../potentials/systems.txt'

# Available systems through potentials
with open(potentials, 'rb') as f:
    potentials = pickle.load(f)

# Johnson data
df = pd.read_csv(inoue)

df['Alloy'] = df['Alloy composition (at.%) [ref.]']
df['Citation'] = df['Alloy'].apply(lambda x: x.strip().split(' ')[-1])
df['Alloy'] = df['Alloy'].apply(lambda x: x.split(' ')[0])

# Create columns of data
df['system'] = df['Alloy'].apply(lambda x: re.findall('[A-Z][a-z]?', x))
df['system'] = df['system'].apply(lambda x: set(x))

# Check mathching potentials
df['potential_match'] = False
for index, row in df.iterrows():

    for system in potentials:
        if row['system'].issubset(system):
            df.at[index, 'potential_match'] = True

dfmatch = df.loc[df['potential_match'] == True]
dfmatch = dfmatch.drop(columns=['potential_match']).reset_index(drop=True)

dfmatch.to_csv('matches.txt', index=False)

# Sort data
dfmatch = dfmatch.sort_values(by=['Dmax\xa0(mm)'])
df = df.sort_values(by=['Dmax\xa0(mm)'])

fig, ax = pl.subplots()

y = dfmatch['Alloy'].values
x = dfmatch['Dmax\xa0(mm)'].values

ax.plot(
        x,
        y,
        linestyle='none',
        marker='8',
        label='Matching Potentials'
        )

ax.legend()
ax.grid()

ax.set_ylabel('Composition')
ax.set_xlabel(r'$D_{max}$ [mm]')

fig.tight_layout()
fig.savefig('choices')

pl.show()
