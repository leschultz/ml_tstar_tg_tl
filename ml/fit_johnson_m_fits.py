from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

# Data path
df = 'data_original/data.txt'
dfmfit = 'data/m_fit.txt'
dfj = 'data_original/johnson_data.txt'

# Import data
df = pd.read_csv(df)
dfmfit = pd.read_csv(dfmfit)
dfj = pd.read_csv(dfj)

# Separte md and experimental data
dfexp = df[df['method'].isin(['experimental'])]
dfmd = df[df['method'].isin(['md'])]

# Truncate data
dfexp = dfexp[['composition', 'tl', 'tg/tl', 'm', 'dmax']]
dfmd = dfmd[['composition', 'tg']]

# Combine fitted m with Trg values from md
dfmfit = pd.merge(dfmfit, dfexp.drop(['m', 'tg/tl'], axis=1), on=['composition'])
dfmfit = pd.merge(dfmfit, dfmd, on=['composition'])
dfmfit['tg/tl'] = dfmfit['tg']/dfmfit['tl']

# Truncate data
dfmfit = dfmfit[['composition', 'tg/tl', 'm', 'dmax']].dropna()
dfj = dfj.dropna()

# Take the log of the squared dmax
dfj['log(dmax^2)'] = np.log10(dfj['dmax']**2)
dfmfit['log(dmax^2)'] = np.log10(dfmfit['dmax']**2)

print(dfj)
print(dfmfit)
# ML
X_train = dfj[['m', 'tg/tl']].values
X_test = dfmfit[['m', 'tg/tl']].values

y_train = dfj['log(dmax^2)'].values
y_test = dfmfit['log(dmax^2)'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_johnson_pred = reg.predict(X_train)
y_mfit_pred = reg.predict(X_test)

# Plots
fig, ax = pl.subplots()

ax.plot(
        y_johnson_pred,
        y_train,
        marker='.',
        linestyle='none',
        label='Original Data'
        )

ax.plot(
        y_mfit_pred,
        y_test,
        marker='8',
        linestyle='none',
        label=r'MD Data'
        )

ax.legend()
ax.grid()

ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

fig.savefig('figures/johnson_fit')

pl.show()
