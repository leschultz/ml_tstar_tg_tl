from pymatgen import Composition, Element
import pandas as pd
import numpy as np


def md_data(dfexp, dfmd, source):

    # Filter by source
    dfexp = dfexp[dfexp['source'].isin([source])]

    # Truncate data
    dfexp = dfexp[['composition', 'tl', 'tg', 'dmax']]
    dfmd = dfmd[['composition', 'tg', 'visc_tcut', 'diff_tcut']]
    dfexp = dfexp.dropna()

    # Change Tg naming convention
    dfexp = dfexp.rename({'tg': 'tg_exp'}, axis=1)
    dfmd = dfmd.rename({'tg': 'tg_md'}, axis=1)

    # Combine use experimental Tl values for MD
    df = pd.merge(
                  dfmd,
                  dfexp,
                  on=['composition']
                  )

    # Save source
    df['source'] = source

    return df


def comp_fracs(x):
    '''
    Calculate the atomic fraction of a composition.

    inputs:
        x = A string of the composition
    outputs:
        x = The composition with atomic fractions
    '''

    x = Composition(x)
    x = x.reduced_formula

    return x


# Data paths
dfexp = '../../../paper_data/data/data.txt'
dfmd = '../../../md/jobs_data/master.csv'

# Import data
dfexp = pd.read_csv(dfexp)
dfmd = pd.read_csv(dfmd)

# Allocate data type
dfexp['method'] = 'experimental'
dfmd['method'] = 'md'

# Reorganize composition naming methodology
dfexp['composition'] = dfexp['composition'].apply(lambda x: x.split(' ')[0])
dfexp['composition'] = dfexp['composition'].apply(lambda x: comp_fracs(x))
dfmd['composition'] = dfmd['composition'].apply(lambda x: comp_fracs(x))

# Combine data
df = pd.concat([dfexp, dfmd], sort=True)

# Calculate both Trg and Tg/T*
df['tg/tl'] = df['tg']/df['tl']
df['tg/visc_tcut'] = df['tg']/df['visc_tcut']
df['tg/diff_tcut'] = df['tg']/df['diff_tcut']

# Separte md and experimental data
dfexp = df[df['method'].isin(['experimental'])]
dfmd = df[df['method'].isin(['md'])]

# Loop for all sources
mean = []
for source in set(df['source'].values):

    # Prevent nan
    if source != source:
        continue

    mean.append(md_data(dfexp, dfmd, source))

df = pd.concat(mean, sort=True)
del mean

df['tg_md/tl'] = df['tg_md']/df['tl']
df['tg_exp/tl'] = df['tg_exp']/df['tl']

df['tg_md/visc_tcut'] = df['tg_md']/df['visc_tcut']
df['tg_md/diff_tcut'] = df['tg_md']/df['diff_tcut']

df['tg_exp/visc_tcut'] = df['tg_exp']/df['visc_tcut']
df['tg_exp/diff_tcut'] = df['tg_exp']/df['diff_tcut']

df[r'$log(dmax^{2})$'] = np.log10(df['dmax']**2)

df.to_csv('../data/data.csv', index=False)
