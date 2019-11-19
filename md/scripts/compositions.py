from pymatgen import Composition

import pandas as pd

import sys
import os

path = sys.argv[1]
save_dir = sys.argv[2]
save_name = sys.argv[3]

folders = os.walk(path)
compositions = []
run_numbers = []
for i in folders:
    composition = i[0].split('/')

    if len(composition) != 3:
        continue

    composition = composition[-1]
    composition = composition.split('_')

    run_number = composition[-1]

    composition = composition[1]
    composition = Composition(composition).fractional_composition*100

    comp = ''
    for key, value in composition.items():

        comp += str(key)+str(int(value))

    compositions.append(comp)
    run_numbers.append(run_number)

df = pd.DataFrame({
                   'composition': compositions,
                   'run': run_numbers
                   })

groups = df.groupby(['composition'])
df = groups.count().add_suffix('_count').reset_index()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df.to_csv(os.path.join(save_dir, save_name), index=False)
