from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

import pickle

# Data path
df = '../data/m_fit.txt'

# Import data
df = pd.read_csv(df)

# Truncate data
df = df.dropna()

# Take the log of the squared dmax
df['log(dmax^2)'] = np.log10(df['dmax']**2)

# ML
X_train = df[['tg_md_mean/tstar_mean', 'tg_md_mean/tl']].values
y_train = df['log(dmax^2)'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Save model
pickle.dump(reg, open('../model/md.sav', 'wb')) 
