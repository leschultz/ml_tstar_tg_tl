from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

# Data paths
dfkelton = '../../paper_data/kelton/m_vs_tgovertstar.txt'
df = '../data_original/data.txt'

# Import data
dfkelton = pd.read_csv(dfkelton)
df = pd.read_csv(df)

# Keep minimum data
df = df[df['method'].isin(['md'])]
df = df[['composition', 'tg/tstar', 'tl', 'tg', 'dmax']]

# Take mean values for each composition
groups = df.groupby(['composition'])
mean = groups.mean().add_suffix('_mean').reset_index()
std = groups.std().add_suffix('_std').reset_index()
sem = groups.sem().add_suffix('_sem').reset_index()
count = groups.count().add_suffix('_count').reset_index()

# Combine Data
df = pd.concat([mean, std, sem, count], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

# Format data for machine learning
compositions = df['composition'].values
X_train = dfkelton[['tg/tstar']].values
X_test = df[['tg/tstar_mean']].values
y_train = dfkelton['m'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Prediction
y_pred = reg.predict(X_test)

# Create dataframe
df = {}
df['composition'] = compositions
df['m'] = y_pred
df = pd.DataFrame(df)
df.to_csv('../data/m_fit.txt', index=False)

# Figures
fig, ax = pl.subplots()

ax.plot(
        dfkelton['tg/tstar'],
        dfkelton['m'],
        marker='.',
        linestyle='none',
        label='Kelton Data'
        )

ax.plot(
        X_test,
        y_pred,
        marker='8',
        linestyle='none',
        label=r'Predicted Data'
        )

ax.legend()
ax.grid()

ax.set_xlabel(r'$T_{g}/T_{*}$ fit [-]')
ax.set_ylabel(r'Fragility Index (m)')

fig.tight_layout()

fig.savefig('../figures/m_vs_tgovertstar')

pl.show()