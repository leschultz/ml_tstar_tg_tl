from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

# Data paths
dfkelton = '../../../paper_data/kelton/m_vs_tgovertstar.txt'
df = '../data/data.csv'

# Import data
dfkelton = pd.read_csv(dfkelton)
df = pd.read_csv(df)

# Remove duplicate data
df = df.drop_duplicates(subset='composition')

# Format data for machine learning
compositions = df['composition'].values
X_train = dfkelton[['tg/tstar']].values
X_mdpure = df[['tg_md/visc_tcut']].values
X_mdpartial = df[['tg_exp/visc_tcut']].values
y_train = dfkelton['m'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
coeffs = reg.coef_

# Prediction
y_mdpure = reg.predict(X_mdpure)
y_mdpartial = reg.predict(X_mdpartial)

# Create dataframe
df['m_md'] = y_mdpure
df['m_exp'] = y_mdpartial
df.to_csv('../data/m_fit.csv', index=False)

# Figures
fig, ax = pl.subplots()

sigs = 6
label = 'Fit: '
label += 'm='
label += str(coeffs[0])[:sigs]+r'$T_{g}/T^{*}$'

ax.set_title(label)

ax.plot(
        dfkelton['tg/tstar'],
        dfkelton['m'],
        marker='.',
        linestyle='none',
        color='k',
        label='Kelton Data'
        )

ax.plot(
        X_mdpure,
        y_mdpure,
        marker='8',
        linestyle='none',
        color='b',
        label=r'Predicted: $T_{g}$ from MD'
        )

ax.plot(
        X_mdpartial,
        y_mdpartial,
        marker='8',
        linestyle='none',
        color='g',
        label=r'Predited: $T_{g}$ from Exp.'
        )

ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
ax.grid()

ax.set_xlabel(r'$T_{g}/T^{*}$ fit [-]')
ax.set_ylabel(r'Fragility Index (m)')

fig.tight_layout()

fig.savefig('../figures/m_vs_tgovertstar')
