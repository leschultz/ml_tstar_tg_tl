from matplotlib import pyplot as pl

from sklearn import linear_model
from sklearn import metrics

import pandas as pd
import numpy as np


def score(r2, mse, mseoversigmay):

    label=r'$R^{2}=$'+str(r2)
    label += '\n'
    label += r'MSE='+str(mse)
    label += '\n'
    label += r'$MSE/\sigma_{y}=$'+str(mseoversigmay)

    return label


# Data paths
dfmd = '../data_original/md_mean.txt'
dfmfit = '../data/m_fit.txt'
dfj = '../data_original/johnson_data.txt'

# Import data
dfmd = pd.read_csv(dfmd)
dfmfit = pd.read_csv(dfmfit)
dfj = pd.read_csv(dfj)

# Combine fitted m with Trg values from md
dfmfit = pd.merge(dfmfit, dfmd)

# Truncate data
dfj = dfj.dropna()
dfmfit = dfmfit.dropna()

# Remove md data from johnson data
dfj = dfj[~dfj['composition'].isin(dfmfit['composition'].values)]

# Calculate features
dfmfit['tg_md/tl'] = dfmfit['tg_md_mean']/dfmfit['tl']
dfmfit['tg_exp/tl'] = dfmfit['tg_exp']/dfmfit['tl']
dfj['tg/tl'] = dfj['tg']/dfj['tl']

# Take the log of the squared dmax
dfj['log(dmax^2)'] = np.log10(dfj['dmax']**2)
dfmfit['log(dmax^2)'] = np.log10(dfmfit['dmax']**2)

# ML
X_train = dfj[['m', 'tg/tl']].values
X_test = dfmfit[['m', 'tg_md/tl']].values

y_train = dfj['log(dmax^2)'].values
y_test = dfmfit['log(dmax^2)'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_johnson_pred = reg.predict(X_train)
y_mfit_pred = reg.predict(X_test)

# ML performance metrics
r2j = metrics.r2_score(y_train, y_johnson_pred)  # R^2
r2mfit = metrics.r2_score(y_test, y_mfit_pred)  # R^2

msej = metrics.mean_squared_error(y_train, y_johnson_pred)  # MSE
msemfit = metrics.mean_squared_error(y_test, y_mfit_pred)  # MSE

mseoversigmayj = msej/np.std(y_train)  # MSE/sigmay
mseoversigmaymfit = msemfit/np.std(y_test)  # MSE/sigmay

# Plots for prediction on training set
fig, ax = pl.subplots()

ax.plot(
        y_johnson_pred,
        y_train,
        marker='.',
        linestyle='none',
        label=score(r2j, msej, mseoversigmayj)
        )

ax.legend()
ax.grid()

ax.set_title('Train')
ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

fig.savefig('../figures/johnson_fit_train')

# Pltos for prediction on testing sets
fig, ax = pl.subplots()

ax.plot(
        y_mfit_pred,
        y_test,
        marker='8',
        linestyle='none',
        label=score(r2mfit, msemfit, mseoversigmaymfit)
        )

ax.legend()
ax.grid()

ax.set_title('Test')
ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

fig.savefig('../figures/johnson_fit_test')

pl.show()
