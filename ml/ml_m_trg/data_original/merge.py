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
dfexp = '../../paper_data/data/data.txt'
dfmd = '../../md/jobs_data/fragility.txt'

# Import data
dfexp = pd.read_csv(dfexp)
dfmd = pd.read_csv(dfmd)

# Allocate data type
dfexp['method'] = 'experimental'
dfmd['method'] = 'md'

# Change md job name and drop job identifier
dfmd['composition'] = dfmd['job'].apply(lambda x: x.split('_')[1])
dfmd = dfmd.drop(['job'], axis=1)

# Reorganize composition naming methodology
dfexp['composition'] = dfexp['composition'].apply(lambda x: x.split(' ')[0])
dfexp['composition'] = dfexp['composition'].apply(lambda x: comp_fracs(x))
dfmd['composition'] = dfmd['composition'].apply(lambda x: comp_fracs(x))

# Combine data
df = pd.concat([dfexp, dfmd], sort=True)

# Calculate both Trg and Tg/T*
df['tg/tl'] = df['tg']/df['tl']
df['tg/tstar'] = df['tg']/df['tstar']

df.to_csv('data.txt', index=False)
