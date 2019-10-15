import pandas as pd

df1 = pd.read_csv('t1.csv')
df2 = pd.read_csv('t2.csv')

df = pd.concat([df1, df2])
df = df.reset_index(drop=True)

df.columns = ['tg/tstar', 'm']

df.to_csv('m_vs_tgovertstar.txt', index=False)
