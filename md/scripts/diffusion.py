from ovito.modifiers import CalculateDisplacementsModifier
from ovito.io import import_file

import pandas as pd
import numpy as np

from functions import finder
from msd import msd, self_diffusion

import os

datadir = '../jobs'
trajfile = 'traj_nvt.lammpstrj'
tempfile = 'temperature.txt'
save = '../jobs_data/diffusions.csv'
timestep = 0.001  # ps
dumprate = 100000
time_modifier = timestep*dumprate

# Find all trajectory files
traj_files = set(finder(trajfile, datadir))

# Find all temperatures
temp_files = set(finder(tempfile, datadir))

if temp_files != traj_files:
    print('Problem with run files')

# Get absolute paths for files
runs = list(traj_files.union(temp_files))  # Union of mathcing paths
traj_files = list(map(lambda x: os.path.join(x, trajfile), runs))
temp_files = list(map(lambda x: os.path.join(x, tempfile), runs))

diffusion = []
temperature = []
good_runs = []
bad_runs = []

msd_modifier = CalculateDisplacementsModifier()
msd_modifier.assume_unwrapped_coordinates = True
for temp, traj in zip(temp_files, traj_files):

    print(traj)

    try:
        temp = np.loadtxt(temp)  # Hold temperature

        pipeline = import_file(traj)
        pipeline.modifiers.append(msd_modifier)
        pipeline.modifiers.append(msd)

        frames = np.array(range(pipeline.source.num_frames))
        time = frames*time_modifier  # Time

        pipeline.modifiers[0].reference_frame = 0  # Origin

        msd_calc = []
        for frame in frames:
            out = pipeline.compute(frame)
            msd_calc.append(out.attributes['msd'])

        diff = self_diffusion(time, msd_calc, 6)

        temperature.append(temp)
        diffusion.append(diff)
        good_runs.append(traj)

    except Exception:
        bad_runs.append(traj)
        pass

df = pd.DataFrame({
                   'run': good_runs,
                   'temperature': temperature,
                   'diffusion': diffusion,
                   })

df.to_csv(save, index=False)

print('Bad runs:')
print(bad_runs)
