from pymatgen import Composition, Element
import pandas as pd


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

df.to_csv('data.txt', index=False)
