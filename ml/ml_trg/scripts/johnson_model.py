from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

import pickle

# Data paths
dfmfit = '../data/m_fit.txt'
dfj = '../data_original/johnson_data.txt'

# Import data
dfmfit = pd.read_csv(dfmfit)
dfj = pd.read_csv(dfj)

# Truncate data
dfj = dfj.dropna()
dfmfit = dfmfit.dropna()

# Remove md data from johnson data
dfj = dfj[~dfj['composition'].isin(dfmfit['composition'].values)]

# Calculate features
dfj['tg/tl'] = dfj['tg']/dfj['tl']

# Take the log of the squared dmax
dfj['log(dmax^2)'] = np.log10(dfj['dmax']**2)

# ML
X_train = dfj[['tg/tl']].values
y_train = dfj['log(dmax^2)'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Save model
pickle.dump(reg, open('../model/johnson.sav', 'wb')) 
