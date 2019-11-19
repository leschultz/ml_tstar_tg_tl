from matplotlib import pyplot as pl

import pandas as pd
import numpy as np
import os

data_dir = '../scores/'

df = []
for i in os.walk(data_dir):
    if 'score' not in i[0]:
        continue

    for j in i[-1]:
        score = pd.read_csv(os.path.join(i[0], j)).set_index('metric').T
        j = float(j.split('.')[0].replace('p', '.'))
        score['cutoff'] = j
        df.append(score)

df = pd.concat(df)

count = df.shape[1]-1
fig, ax = pl.subplots(count)

ax[0].plot(df['cutoff'], df['r2'], marker='8', linestyle='none')
ax[1].plot(df['cutoff'], df['mse'], marker='8', linestyle='none')
ax[2].plot(df['cutoff'], df['mseoversigmay'], marker='8', linestyle='none')

ax[0].set_ylabel(r'$R^{2}$')
ax[1].set_ylabel(r'$MSE$')
ax[2].set_ylabel(r'$MSE/\sigma_{y}$')

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[-1].set_xlabel(r'Viscosity Cutoff $[Pa \cdot s]$')

fig.savefig('../plots/cutoff_scores')

pl.show()
