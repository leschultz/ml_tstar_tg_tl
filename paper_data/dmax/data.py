import pandas as pd

dfi = pd.read_csv('../inoue/matches.txt')
dfj = pd.read_csv('../johnson/matches.txt')
dfw = pd.read_csv('../ward/matches.txt')

# Gather only relevant columns
dfi = dfi[['Alloy', 'Tg\xa0(K)', 'Tl\xa0(K)', 'Dmax\xa0(mm)']]
dfj = dfj[['Alloy', 'Tg (K)', 'TL (K)', 'dexp (mm)']]
dfw = dfw[['composition', 'D_max']]

# Standardize columns
dfi.columns = ['alloy', 'tg', 'tl', 'dmax']
dfj.columns = ['alloy', 'tg', 'tl', 'dmax']
dfw.columns = ['alloy', 'dmax']

# State source of data
dfi['source'] = 'inoue'
dfj['source'] = 'johnson'
dfw['source'] = 'ward'

# Combine available data
df = pd.concat([dfi, dfj, dfw], sort=True)
df.to_csv('data.txt', index=False)
