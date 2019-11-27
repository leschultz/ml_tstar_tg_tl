from matplotlib import pyplot as pl

from sklearn import linear_model
from sklearn import metrics

import pandas as pd
import numpy as np

import pickle


def score(label, r2, mse, mseoversigmay, digits):

    r2 = str(round(r2, digits))
    mse = str(round(mse, digits))
    mseoversigmay = str(round(mseoversigmay, digits))

    label += '\n'
    label += r'$R^{2}=$'+r2
    label += '\n'
    label += r'MSE='+mse
    label += '\n'
    label += r'$MSE/\sigma_{y}=$'+mseoversigmay

    return label


def ml(reg, X_train, y_train):

    y_pred = reg.predict(X_train)  # Predictions
    r2 = metrics.r2_score(y_train, y_pred)  # R^2
    mse = metrics.mean_squared_error(y_train, y_pred)  # MSE
    mseoversigmay = mse/np.std(y_train)  # MSE/sigmay

    return y_pred, r2, mse, mseoversigmay


# Data paths
dfmfit = '../data/m_fit.csv'
dfj = '../data/data_johnson.csv'

# Import data
dfmfit = pd.read_csv(dfmfit)
dfj = pd.read_csv(dfj)

# Truncate data
dfj = dfj.dropna()

# Calculate features
dfj['tg/tl'] = dfj['tg']/dfj['tl']

# ML
X_johnson = dfj[['m', 'tg/tl']].values
y_johnson = dfj[r'$log(dmax^{2})$'].values

X_mdpure = dfmfit[['m_md', 'tg_md/tl']].values
X_mdpartial = dfmfit[['m_exp', 'tg_exp/tl']].values
y_md = dfmfit[r'$log(dmax^{2})$'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_johnson, y_johnson)
coeffs = reg.coef_

# Predictions
predj = ml(reg, X_johnson, y_johnson)
predmdpure = ml(reg, X_mdpure, y_md)
predmdpartial = ml(reg, X_mdpartial, y_md)

predj, r2j, msej, mseoversigmayj = predj
predmdpure, r2mdpure, msemdpure, mseoversigmaymdpure = predmdpure
predmdpartial, r2mdpartial, msemdpartial, mseoversigmaymdpartial = predmdpartial

dfj['log(dmax^2)_pred'] = predj
dfmfit['log(dmax^2)_tg_md_pred'] = predmdpure
dfmfit['log(dmax^2)_tg_exp_pred'] = predmdpartial

# Save performance scores
score_type = ['r2', 'mse', 'mseoversigmay']
score_value = [r2j, msej, mseoversigmayj]
dfj_score = pd.DataFrame({
                          'metric': score_type,
                          'score': score_value
                          })

score_value = [r2mdpure, msemdpure, mseoversigmaymdpure]
dfmdpure_score = pd.DataFrame({
                               'metric': score_type,
                               'score': score_value
                               })

score_value = [r2mdpartial, msemdpartial, mseoversigmaymdpartial]
dfmdpartial_score = pd.DataFrame({
                                  'metric': score_type,
                                  'score': score_value
                                  })

# Saving data
dfj_score.to_csv('../data/johson_model_johnson_pred_score.csv', index=False)
dfmdpure_score.to_csv('../data/johson_model_mdpure_pred_score.csv', index=False)
dfmdpartial_score.to_csv('../data/johson_model_mdpartial_pred_score.csv', index=False)

dfj.to_csv('../data/johson_model_johnson_pred.csv', index=False)
dfmfit.to_csv('../data/johnson_model_md_pred.csv', index=False)

# Plots for prediction on training set
fig, ax = pl.subplots()

sigs = 3
label = 'Fit: '
label += r'log($dmax^2$)='+str(round(coeffs[0], sigs))+'m+'
label += str(round(coeffs[1], sigs))+r'$T_{rg}$'

ax.set_title(label)
ax.plot(
        predj,
        y_johnson,
        marker='.',
        linestyle='none',
        color='k',
        label=score('Johnson Exp', r2j, msej, mseoversigmayj, sigs)
        )

# Plots for prediction on testing sets
ax.plot(
        predmdpure,
        y_md,
        marker='*',
        linestyle='none',
        color='b',
        label=score(r'$T_{g}$ MD', r2mdpure, msemdpure, mseoversigmaymdpure, sigs)
        )

# Plots for prediction on testing sets
ax.plot(
        predmdpartial,
        y_md,
        marker='8',
        linestyle='none',
        color='g',
        label=score(r'$T_{g}$ Exp', r2mdpartial, msemdpartial, mseoversigmaymdpartial, sigs)
        )

ax.set_aspect('equal', 'box')
ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
ax.grid()

ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

# Label points on graph
for i, j, k, l in zip(dfmfit['composition'], predmdpure, predmdpartial, y_md):
    ax.annotate(i, (j, l))
    ax.annotate(i, (k, l))

fig.savefig('../figures/johnson_fit')
