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


# Data path
df = '../data/data.csv'

# Import data
df = pd.read_csv(df)
df = df[~df['composition'].isin(['Nb2Ni3'])]

# Drop duplicate compositions
df = df.drop_duplicates(subset='composition', keep='first')

# ML
X_train = df[['tg_md/tl']].values
y_train = df[r'$log(dmax^{2})$'].values

# Model
reg = linear_model.LinearRegression()  # Import model
reg.fit(X_train, y_train)  # Train the model
coeffs = reg.coef_  # Fitting coefficients
intercpet = reg.intercept_  # Intercept

# Predictions
pred = ml(reg, X_train, y_train)
y_pred, r2, mse, mseoversigmay = pred

# Save performance scores
score_type = ['r2', 'mse', 'mseoversigmay']
score_value = [r2, mse, mseoversigmay]
df_score = pd.DataFrame({
                         'metric': score_type,
                         'score': score_value
                         })

# Saving data
df_score.to_csv('../data/md_score_trg.csv', index=False)
df['log(dmax^2)_pred'] = y_pred
df.to_csv('../data/md_trg.csv', index=False)

# Plots for prediction on testing sets
fig, ax = pl.subplots()

sigs = 3
label = 'MD Fit: '
label += r'log($dmax^2$)='
label += str(round(coeffs[0], sigs))+r'$T_{rg}$'
label += '+('+str(round(intercpet, sigs))+')'

ax.set_title(label)

ax.plot(
        y_pred,
        y_train,
        marker='*',
        linestyle='none',
        color='b',
        label=score(r'$T_{g}$ MD', r2, mse, mseoversigmay, sigs)
        )

ax.legend(loc='best')
ax.grid()

ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

if ax.get_ylim()[1] > ax.get_xlim()[1]:
    max_lim = ax.get_ylim()[1]
else:
    max_lim = ax.get_xlim()[1]

if ax.get_ylim()[0] < ax.get_xlim()[0]:
    min_lim = ax.get_ylim()[0]
else:
    min_lim = ax.get_xlim()[0]

ax.set_xlim((min_lim, max_lim))
ax.set_ylim((min_lim, max_lim))
ax.set_aspect('equal')

# Label points on graph
for i, j, l in zip(df['composition'], y_pred, y_train):
    ax.annotate(i, (j, l))

fig.savefig('../figures/md_trg_fit')
