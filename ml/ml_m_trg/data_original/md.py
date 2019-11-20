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


# Load Data
df = pd.read_csv('data.txt')

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

df.to_csv('md_mean.txt', index=False)

print(df)
