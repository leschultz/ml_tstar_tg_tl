from matplotlib import pyplot as pl

from sklearn import linear_model
from sklearn import metrics

import pandas as pd
import numpy as np

import pickle


def score(label, r2, mse, mseoversigmay):

    label += '\n'
    label += r'$R^{2}=$'+str(r2)
    label += '\n'
    label += r'MSE='+str(mse)
    label += '\n'
    label += r'$MSE/\sigma_{y}=$'+str(mseoversigmay)

    return label


def ml(reg, X_train, y_train):

    y_pred = reg.predict(X_train)  # Predictions
    r2 = metrics.r2_score(y_train, y_pred)  # R^2
    mse = metrics.mean_squared_error(y_train, y_pred)  # MSE
    mseoversigmay = mse/np.std(y_train)  # MSE/sigmay

    return y_pred, r2, mse, mseoversigmay


# Data paths
dfmfit = '../data/m_fit.txt'
dfj = '../data_original/johnson_data.txt'

# Import data
dfmfit = pd.read_csv(dfmfit)
dfj = pd.read_csv(dfj)

# Truncate data
dfj = dfj.dropna()

# Calculate features
dfmfit['tg_md/tl'] = dfmfit['tg_md_mean']/dfmfit['tl']
dfmfit['tg_exp/tl'] = dfmfit['tg_exp']/dfmfit['tl']
dfj['tg/tl'] = dfj['tg']/dfj['tl']

# Take the log of the squared dmax
dfj['log(dmax^2)'] = np.log10(dfj['dmax']**2)
dfmfit['log(dmax^2)'] = np.log10(dfmfit['dmax']**2)

# ML
X_johnson = dfj[['m']].values
y_johnson = dfj['log(dmax^2)'].values

X_mdpure = dfmfit[['m_md']].values
X_mdpartial = dfmfit[['m_exp']].values
y_md = dfmfit['log(dmax^2)'].values

# Model
reg = pickle.load(open('../model/johnson.sav', 'rb'))

# Predictions
predj = ml(reg, X_johnson, y_johnson)
predmdpure = ml(reg, X_mdpure, y_md)
predmdpartial =  ml(reg, X_mdpartial, y_md)

predj, r2j, msej, mseoversigmayj = predj
predmdpure, r2mdpure, msemdpure, mseoversigmaymdpure = predmdpure
predmdpartial, r2mdpartial, msemdpartial, mseoversigmaymdpartial = predmdpartial

# Plots for prediction on training set
fig, ax = pl.subplots()

ax.plot(
        predj,
        y_johnson,
        marker='.',
        linestyle='none',
        label=score('Johnson Data', r2j, msej, mseoversigmayj)
        )

# Plots for prediction on testing sets
ax.plot(
        predmdpure,
        y_md,
        marker='*',
        linestyle='none',
        label=score(r'$T_{g}$ MD', r2mdpure, msemdpure, mseoversigmaymdpure)
        )

# Plots for prediction on testing sets
ax.plot(
        predmdpartial,
        y_md,
        marker='8',
        linestyle='none',
        label=score(r'$T_{g}$ Exp', r2mdpartial, msemdpartial, mseoversigmaymdpartial)
        )

ax.legend()
ax.grid()

ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

# Label points on graph
for i, j, k, l in zip(dfmfit['composition'], predmdpure, predmdpartial, y_md):
    ax.annotate(i, (j, l))
    ax.annotate(i, (k, l))

pl.show()

fig.savefig('../figures/johnson_fit')
