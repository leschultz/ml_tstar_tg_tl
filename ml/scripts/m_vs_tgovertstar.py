from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

# Data paths
dfkelton = '../../paper_data/kelton/m_vs_tgovertstar.txt'
df = '../data_original/md_mean.txt'

# Import data
dfkelton = pd.read_csv(dfkelton)
df = pd.read_csv(df)

# Format data for machine learning
compositions = df['composition'].values
X_train = dfkelton[['tg/tstar']].values
X_mdpure = df[['tg_md_mean/tstar_mean']].values
X_mdpartial = df[['tg_exp/tstar_mean']].values
y_train = dfkelton['m'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Prediction
y_mdpure = reg.predict(X_mdpure)
y_mdpartial = reg.predict(X_mdpartial)

# Create dataframe
df['m_md'] = y_mdpure
df['m_exp'] = y_mdpartial
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
        X_mdpure,
        y_mdpure,
        marker='8',
        linestyle='none',
        label=r'Predicted: $T_{g}$ from MD'
        )

ax.plot(
        X_mdpartial,
        y_mdpartial,
        marker='8',
        linestyle='none',
        label=r'Predited: $T_{g}$ from Exp.'
        )

ax.legend()
ax.grid()

ax.set_xlabel(r'$T_{g}/T_{*}$ fit [-]')
ax.set_ylabel(r'Fragility Index (m)')

fig.tight_layout()

fig.savefig('../figures/m_vs_tgovertstar')

pl.show()
