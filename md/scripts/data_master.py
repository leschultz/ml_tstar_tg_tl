import pandas as pd

df_tcuts = pd.read_csv('../jobs_data/tcuts.csv')
df_tg = pd.read_csv('../jobs_data/tg.txt')

df_tg['composition'] = df_tg['job'].apply(lambda x: x.split('_')[1])
df_tg = df_tg.drop(['job'], axis=1)

groups = df_tg.groupby(['composition'])

df_tg = groups.mean().reset_index()

df = pd.merge(df_tg, df_tcuts, on=['composition'])

df.to_csv('../jobs_data/master.csv', index=False)
