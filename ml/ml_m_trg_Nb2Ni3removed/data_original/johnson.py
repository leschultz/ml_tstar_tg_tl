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


# Data path
df = '../../../paper_data/johnson/data.txt'

# Import data
df = pd.read_csv(df)

# Truncate data
df = df[['Alloy', 'Tg (K)', 'TL (K)', 'm', 'dexp (mm)']]

# Change column names
df.columns = ['composition', 'tg', 'tl', 'm', 'dmax']

# Reorganize composition naming methodology
df['composition'] = df['composition'].apply(lambda x: x.split(' ')[0])
df['composition'] = df['composition'].apply(lambda x: comp_fracs(x))

# Calculate both Trg and Tg/T*
df['tg/tl'] = df['tg']/df['tl']

df.to_csv('johnson_data.txt', index=False)
