from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

# Data path
df = '../data/m_fit.txt'

# Import data
df = pd.read_csv(df)

fig, ax = pl.subplots()

x = df['tg_md_mean/tstar_mean']
y = df['dmax']
xerr = df['tg_md_mean/tstar_mean_err']

ax.errorbar(
            x,
            y,
            xerr=xerr,
            ecolor='r',
            marker='8',
            linestyle='none',
            )

ax.grid()

ax.set_xlabel(r'$T_{g}/T^{*}$')
ax.set_ylabel(r'$D_{max}$ $[mm]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../figures/dmax_vs_tgovertstar')

pl.show()
