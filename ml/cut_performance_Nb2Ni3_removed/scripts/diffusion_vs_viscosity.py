from sklearn import linear_model
from sklearn import metrics

from functions import *
import pandas as pd
import numpy as np

visc_cuts = np.linspace(0.01, 1, 100)
diff_cuts = np.linspace(0.0001, 0.3, 100)
density = 10000

# Data paths
df_visc = pd.read_csv('/home/leschultz/work/tstar/md/jobs_data/viscosities.txt')
df_diff = pd.read_csv('/home/leschultz/work/tstar/md/jobs_data/diffusions.csv')
df_tg = pd.read_csv('/home/leschultz/work/tstar/md/jobs_data/tg.txt')
dfexp = pd.read_csv('/home/leschultz/work/tstar/paper_data/data/data.txt')

dfmd = md_prep(df_visc, df_diff, df_tg)

# Reorganize composition naming methodology
dfexp['composition'] = dfexp['composition'].apply(lambda x: x.split(' ')[0])
dfexp['composition'] = dfexp['composition'].apply(lambda x: comp_fracs(x))
dfmd['composition'] = dfmd['composition'].apply(lambda x: comp_fracs(x))

dfexp = dfexp[['composition', 'tl', 'dmax']]
dfexp = dfexp.dropna()

dfmd = dfmd[~dfmd['composition'].isin(['Nb2Ni3'])]

groups = dfmd.groupby(['composition'])

data = []
for diff_cut in diff_cuts:

    composition, diff_tcut, visc_tcut, tgs = cuts(
                                                  groups,
                                                  diff_cut,
                                                  visc_cuts[0],
                                                  density
                                                  )

    dfcuts = pd.DataFrame({
                           'composition': composition,
                           'diff_tcut': diff_tcut,
                           'visc_tcut': visc_tcut,
                           'tg': tgs,
                           })

    # Combine data
    df = dfexp.merge(dfcuts, on=['composition'])

    # Drop duplicate compositions
    df = df.drop_duplicates(subset='composition', keep='first')

    # Calculate both Trg and Tg/T*
    df['tg/tl'] = df['tg']/df['tl']
    df['tg/visc_tcut'] = df['tg']/df['visc_tcut']
    df['tg/diff_tcut'] = df['tg']/df['diff_tcut']


    y_train = df['dmax'].values

    X_train_diff = df[['tg/tl', 'tg/diff_tcut']].values
    X_train_visc = df[['tg/tl', 'tg/visc_tcut']].values


    # Model
    reg_diff = linear_model.LinearRegression()
    reg_visc = linear_model.LinearRegression()
    
    reg_diff.fit(X_train_diff, y_train)
    reg_visc.fit(X_train_visc, y_train)

    diff_pred = ml(reg_diff, X_train_diff, y_train)
    visc_pred = ml(reg_diff, X_train_visc, y_train)

    data.append({
                 'diff_cut': diff_cut,
                 'diff_cut_scores': diff_pred
                 })

fig_diff, ax_diff = pl.subplots(3)
for i in data:

    diff_cut = i['diff_cut']
    diff_pred = i['diff_cut_scores']

    y_pred_diff, r2_diff, mse_diff, mseoversigmay_diff = diff_pred

    ax_diff[0].plot(diff_cut, r2_diff, marker='.', linestyle='none', color='b')
    ax_diff[1].plot(diff_cut, mse_diff, marker='.', linestyle='none', color='b')
    ax_diff[2].plot(diff_cut, mseoversigmay_diff, marker='.', linestyle='none', color='b')

    ax_diff[0].set_ylabel(r'$R^{2}$')
    ax_diff[1].set_ylabel(r'$MSE$')
    ax_diff[2].set_ylabel(r'$MSE/\sigma_{y}$')

    ax_diff[0].grid()
    ax_diff[1].grid()
    ax_diff[2].grid()

    ax_diff[-1].set_xlabel(r'Diffusion Cutoff $[*10^{-4} cm^{2} s^{-1}]$')

fig_diff.savefig('../plots/diff_cut')

# Same code except for viscosity
data = []
for visc_cut in visc_cuts:

    composition, diff_tcut, visc_tcut, tgs = cuts(
                                                  groups,
                                                  diff_cuts[0],
                                                  visc_cut,
                                                  density
                                                  )

    dfcuts = pd.DataFrame({
                           'composition': composition,
                           'diff_tcut': diff_tcut,
                           'visc_tcut': visc_tcut,
                           'tg': tgs,
                           })

    # Combine data
    df = dfexp.merge(dfcuts, on=['composition'])

    # Calculate both Trg and Tg/T*
    df['tg/tl'] = df['tg']/df['tl']
    df['tg/visc_tcut'] = df['tg']/df['visc_tcut']
    df['tg/diff_tcut'] = df['tg']/df['diff_tcut']


    y_train = df['dmax'].values

    X_train_diff = df[['tg/tl', 'tg/diff_tcut']].values
    X_train_visc = df[['tg/tl', 'tg/visc_tcut']].values


    # Model
    reg_diff = linear_model.LinearRegression()
    reg_visc = linear_model.LinearRegression()

    reg_diff.fit(X_train_diff, y_train)
    reg_visc.fit(X_train_visc, y_train)

    diff_pred = ml(reg_diff, X_train_diff, y_train)
    visc_pred = ml(reg_diff, X_train_visc, y_train)

    data.append({
                 'visc_cut': visc_cut,
                 'visc_cut_scores': visc_pred
                 })


fig_diff, ax_diff = pl.subplots(3)
for i in data:

    diff_cut = i['visc_cut']
    diff_pred = i['visc_cut_scores']

    y_pred_diff, r2_diff, mse_diff, mseoversigmay_diff = diff_pred

    ax_diff[0].plot(diff_cut, r2_diff, marker='.', linestyle='none', color='b')
    ax_diff[1].plot(diff_cut, mse_diff, marker='.', linestyle='none', color='b')
    ax_diff[2].plot(diff_cut, mseoversigmay_diff, marker='.', linestyle='none', color='b')

    ax_diff[0].set_ylabel(r'$R^{2}$')
    ax_diff[1].set_ylabel(r'$MSE$')
    ax_diff[2].set_ylabel(r'$MSE/\sigma_{y}$')

    ax_diff[0].grid()
    ax_diff[1].grid()
    ax_diff[2].grid()

    ax_diff[-1].set_xlabel(r'Viscosity Cutoff $[Pa \cdot s]$')

fig_diff.savefig('../plots/visc_cut')
