import pandas as pd
import numpy as np


def uncertainty(x, y, z):

    '''
    Propagate uncertaintites for division or multiplication.

    inputs:
        x = Measured values
        y = Measured values uncertatinties
        z = Final measured value from calculation

    outputs:
        err = The uncertainty
    '''
    
    print(x)
    print(y)
    err = 0.0
    for i, j in zip(x, y):
        err += (j/x)**2

    err **= 0.5
    err *= abs(z)

    return err


def md_data(dfexp, dfmd, source):

    # Filter by source
    dfexp = dfexp[dfexp['source'].isin([source])]

    # Truncate data
    dfexp = dfexp[['composition', 'tl', 'tg', 'dmax']]
    dfmd = dfmd[['composition', 'tg', 'tstar']]

    # Take mean experimental values
    dfexp = dfexp.groupby(['composition']).mean()
    dfexp = dfexp.dropna()

    # Change Tg naming convention
    dfexp = dfexp.rename({'tg': 'tg_exp'}, axis=1)
    dfmd = dfmd.rename({'tg': 'tg_md'}, axis=1)

    # Take statistic values for each composition
    groups = dfmd.groupby(['composition'])
    mean = groups.mean().add_suffix('_mean').reset_index()
    std = groups.std().add_suffix('_std').reset_index()
    sem = groups.sem().add_suffix('_sem').reset_index()
    count = groups.count().add_suffix('_count').reset_index()

    # Combine statistics
    mean = mean.merge(std)
    mean = mean.merge(sem)
    mean = mean.merge(count)

    # Combine use experimental Tl values for MD
    mean = pd.merge(
                    mean,
                    dfexp,
                    on=['composition']
                    )

    # Save source
    mean['source'] = source

    return mean


# Load Data
df = pd.read_csv('data.txt')

# Separte md and experimental data
dfexp = df[df['method'].isin(['experimental'])]
dfmd = df[df['method'].isin(['md'])]

# Loop for all sources
mean = []
for source in set(df['source'].values):

    if source != source:
        continue

    mean.append(md_data(dfexp, dfmd, source))

mean = pd.concat(mean, sort=True)

# Calculate features
mean['tg_md_mean/tl'] = mean['tg_md_mean']/mean['tl']
mean['tg_exp/tl'] = mean['tg_exp']/mean['tl']
mean['tg_md_mean/tstar_mean'] = mean['tg_md_mean']/mean['tstar_mean']
mean['tg_exp/tstar_mean'] = mean['tg_exp']/mean['tstar_mean']

# Feature uncertainties
mean['tg_md_mean/tl_err'] = mean['tg_md_mean/tl']*((mean['tg_md_sem']/mean['tg_md_mean'])**2)**0.5
mean['tg_md_mean/tstar_mean_err'] = mean['tg_md_mean/tstar_mean']*((mean['tg_md_sem']/mean['tg_md_mean'])**2+(mean['tstar_sem']/mean['tstar_mean'])**2)**0.5
mean['tg_exp/tstar_mean_err'] = mean['tg_md_mean/tl']*((mean['tstar_sem']/mean['tstar_mean'])**2)**0.5

mean.to_csv('md_mean.txt', index=False)
