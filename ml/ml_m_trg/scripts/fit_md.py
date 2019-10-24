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


# Data path
dfmfit = '../data/m_fit.txt'

# Model
reg = pickle.load(open('../model/md.sav', 'rb'))
coeffs = reg.coef_

# Import data
dfmfit = pd.read_csv(dfmfit)

# Truncate data

# Calculate features
dfmfit['tg_md/tl'] = dfmfit['tg_md_mean']/dfmfit['tl']
dfmfit['tg_exp/tl'] = dfmfit['tg_exp']/dfmfit['tl']

# Take the log of the squared dmax
dfmfit['log(dmax^2)'] = np.log10(dfmfit['dmax']**2)

# ML
X_mdpure = dfmfit[['tg_md_mean/tstar_mean', 'tg_md/tl']].values
X_mdpartial = dfmfit[['tg_exp/tstar_mean', 'tg_exp/tl']].values
y_md = dfmfit['log(dmax^2)'].values

# Predictions
predmdpure = ml(reg, X_mdpure, y_md)
predmdpartial =  ml(reg, X_mdpartial, y_md)

predmdpure, r2mdpure, msemdpure, mseoversigmaymdpure = predmdpure
predmdpartial, r2mdpartial, msemdpartial, mseoversigmaymdpartial = predmdpartial

dfmfit['log(dmax^2)_tg_md_pred'] = predmdpure
dfmfit['log(dmax^2)_tg_exp_pred'] = predmdpartial

dfmfit.to_csv('../data/md_model_md_pred.csv', index=False)

# Plots for prediction on testing sets
fig, ax = pl.subplots()

sigs = 6
label = 'MD Fit: '
label += 'log(dmax^2)='
label += str(coeffs[0])[:sigs]+'m+'+str(coeffs[1])[:sigs]+r'$T_{rg}$'

ax.set_title(label)

ax.plot(
        predmdpure,
        y_md,
        marker='*',
        linestyle='none',
        color='b',
        label=score(r'$T_{g}$ MD', r2mdpure, msemdpure, mseoversigmaymdpure)
        )

# Plots for prediction on testing sets
ax.plot(
        predmdpartial,
        y_md,
        marker='8',
        linestyle='none',
        color='g',
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

fig.savefig('../figures/md_fit')
