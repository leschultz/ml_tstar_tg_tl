import pandas as pd


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

    # Combine use experimental Tl values for MD
    dfmd = pd.merge(
                    dfmd,
                    dfexp,
                    on=['composition']
                    )

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

mean.to_csv('md_mean.txt', index=False)
