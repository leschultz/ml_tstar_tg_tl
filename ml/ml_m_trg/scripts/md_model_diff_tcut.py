from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

import pickle

# Data path
df = '../data_original/md_mean.txt'

# Import data
df = pd.read_csv(df)

# Truncate data
df = df.dropna()

# ML
X_train = df[['tg_md/diff_tcut', 'tg_md/tl']].values
y_train = df[r'$log(dmax^{2})$'].values

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Save model
pickle.dump(reg, open('../model/md_diff_tcut.sav', 'wb')) 
