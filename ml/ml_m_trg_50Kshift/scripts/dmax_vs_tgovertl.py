from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

# Data path
df = '../data/m_fit.txt'

# Import data
df = pd.read_csv(df)

fig, ax = pl.subplots()

x = df['tg_md_mean/tl']
y = df['dmax']
xerr = df['tg_md_mean/tl_err']

ax.errorbar(
            x,
            y,
            xerr=xerr,
            ecolor='r',
            marker='8',
            linestyle='none',
            )

ax.grid()

ax.set_xlabel(r'$T_{g}/T_{l}$')
ax.set_ylabel(r'$D_{max}$ $[mm]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../figures/dmax_vs_tgovertl')

pl.show()
