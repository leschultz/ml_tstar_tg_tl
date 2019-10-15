from matplotlib import pyplot as pl

import pandas as pd
import numpy as np

import pickle
import json
import os
import re

potentials = '../potentials/systems.txt'
ward = 'bmg_data.json'

# Available systems through potentials
with open(potentials, 'rb') as f:
    potentials = pickle.load(f)

# Data from Logan Ward
df = json.load(open(ward, 'r'))

# Create columns of data
df = pd.DataFrame(df)
df = pd.concat([df, df['data'].apply(pd.Series)], axis=1)
df['system'] = df['composition'].apply(lambda x: re.findall('[A-Z][a-z]?', x))
df['system'] = df['system'].apply(lambda x: set(x))
df = df.dropna(subset=['D_max'])  # Remove missing Dmax values
df = df.sort_values(by=['D_max'])  # Sort

# Check mathching potentials
df['potential_match'] = False
for index, row in df.iterrows():

    for system in potentials:
        if row['system'].issubset(system):
            df.at[index, 'potential_match'] = True

dfmatch = df.loc[df['potential_match'] == True]
dfmatch = dfmatch.drop(columns=['potential_match']).reset_index(drop=True)

dfmatch.to_csv('matches.txt', index=False)

fig, ax = pl.subplots()

y = dfmatch['composition'].values
x = dfmatch['D_max'].values

ax.plot(
        x,
        y,
        linestyle='none',
        marker='.',
        label='Matching Potentials'
        )

ax.legend()
ax.grid()

ax.set_ylabel('Composition')
ax.set_xlabel(r'$D_{max}$ [mm]')

fig.tight_layout()
fig.savefig('choices')

pl.show()
