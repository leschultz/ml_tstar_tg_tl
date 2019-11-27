from matplotlib import pyplot as pl
from scipy import interpolate
import pandas as pd
import numpy as np

visc_cut = 0.075
diff_cut = 0.1
density = 10000

df_visc = pd.read_csv('../jobs_data/viscosities.txt')
df_diff = pd.read_csv('../jobs_data/diffusions.csv')
df_tg = pd.read_csv('../jobs_data/tg.txt')


def find_nearest(array, value):
    '''
    Find the nearest point.

    inputs:
        array = The data array.
        value = The point in question.

    outputs:
        idx = The index of the nearest value.
    '''

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


df_visc['run'] = df_visc['run'].apply(lambda x: x.split('/')[0])
df_diff['run'] = df_diff['run'].apply(lambda x: x.split('/')[-4])

df_visc.columns = ['run', 'temperature', 'viscosity']
df_tg.columns = ['run', 'tg']

df = df_diff.merge(df_visc, on=['run', 'temperature'], how='left')
df = df.merge(df_tg, on=['run'], how='left')

df['composition'] = df['run'].apply(lambda x: x.split('_')[1])

df = df.drop(['run'], axis=1)

df[r'$ln(D)$'] = np.log(df['diffusion'])
df[r'$ln(\mu)$'] = np.log(df['viscosity'])

groups = df.groupby(['composition', 'temperature'])

mean = groups.mean().add_suffix('_mean').reset_index()
sem = groups.sem().add_suffix('_sem').reset_index()
count = groups.count().add_suffix('_count').reset_index()

df = mean.merge(sem)
df = df.merge(count)

df = df.sort_values(by=['temperature'])

groups = df.groupby(['composition'])

diff_hline = np.log(diff_cut)
visc_hline = np.log(visc_cut)

# Figures
fig, ax = pl.subplots()
fig_diff, ax_diff = pl.subplots()
fig_visc, ax_visc = pl.subplots()

# Cut offs
ax_diff.axhline(diff_hline, linestyle=':', color='k', label='cutoff')
ax_visc.axhline(visc_hline, linestyle=':', color='k', label='cutoff')

# Data Colection
composition = []
diff_tcut = []
visc_tcut = []
tgs = []
for group, values in groups:

    tg = values['tg_mean'].values[0]  # Should be unique
    temp = values['temperature'].values
    x = tg/temp
    y = values[r'$ln(\mu)$_mean'].values
    z = values[r'$ln(D)$_mean'].values
    yovertemp = y-np.log10(temp)

    ynew = np.linspace(min(y), max(y), density)
    xnew = np.linspace(min(x), max(x), density)
    yovertempnew = np.linspace(min(yovertemp), max(yovertemp), density)

    fxy = interpolate.interp1d(x, y)
    fxz = interpolate.interp1d(x, z)
    fyz = interpolate.interp1d(yovertemp, z)

    znew = fyz(yovertempnew)

    ax.plot(
            yovertempnew,
            znew,
            label=group
            )

    ynew = fxy(xnew)
    visc_ind = find_nearest(ynew, visc_hline)

    ax_visc.plot(
                 xnew,
                 ynew,
                 label=group
                 )

    ax_visc.plot(xnew[visc_ind], visc_hline, marker='.', color='k')

    znew = fxz(xnew)
    diff_ind = find_nearest(znew, diff_hline)

    ax_diff.plot(
                 xnew,
                 znew,
                 label=group
                 )

    ax_diff.plot(xnew[diff_ind], diff_hline, marker='.', color='k')

    composition.append(group)
    diff_tcut.append(tg*(xnew[diff_ind])**-1)
    visc_tcut.append(tg*(xnew[visc_ind])**-1)
    tgs.append(tg)

ax.set_xlabel(r'$ln(\mu/Temperature)$ $[ln(Pa \cdot s)]$')
ax.set_ylabel(r'ln(D) [$ln(*10^-4 cm^{2} s^{-1})$]')

ax_diff.set_xlabel(r'$T_{g}/Temperature$ $[K^{-1}]$')
ax_diff.set_ylabel(r'ln(D) [$ln(*10^-4 cm^{2} s^{-1})$]')

ax_visc.set_xlabel(r'$T_{g}/Temperature$ $[K^{-1}]$')
ax_visc.set_ylabel(r'$ln(\mu)$ $[ln(Pa \cdot s)]$')

ax.legend()
ax_diff.legend()
ax_visc.legend()

fig.tight_layout()
fig_diff.tight_layout()
fig_visc.tight_layout()

fig.savefig('../jobs_plots/diffusion_vs_viscosity.png')
fig_diff.savefig('../jobs_plots/diffusion_vs_temperature.png')
fig_visc.savefig('../jobs_plots/viscosity_vs_temperature.png')

df = pd.DataFrame({
                   'composition': composition,
                   'diff_tcut': diff_tcut,
                   'visc_tcut': visc_tcut,
                   'tg': tgs,
                   })

df.to_csv('../jobs_data/master.csv', index=False)

print(df)
