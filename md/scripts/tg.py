from matplotlib import pyplot as pl

from scipy.constants import physical_constants
from scipy.interpolate import UnivariateSpline
from functions import finder, data_parse
from scipy.stats import linregress
from sklearn.cluster import KMeans
from knee import opt

import pandas as pd
import scipy as sc
import numpy as np
import argparse
import os
import re


parser = argparse.ArgumentParser(
                                 description='Arguments for gathering Tg.'
                                 )

parser.add_argument(
                    '-d',
                    action='store',
                    type=str,
                    help='The location of data.'
                    )

parser.add_argument(
                    '-n',
                    action='store',
                    type=str,
                    help='The name of data files.'
                    )

parser.add_argument(
                    '-t',
                    action='store',
                    type=str,
                    help='The file containing the hold temperature.'
                    )

parser.add_argument(
                    '-k',
                    action='store',
                    type=int,
                    help='The number of K-Means clusters.'
                    )

parser.add_argument(
                    '-s',
                    action='store',
                    type=int,
                    help='The number of iterations to use for Tg values.'
                    )

parser.add_argument(
                    '-v',
                    action='store',
                    type=int,
                    help='The number of points to use in linear interpolation.'
                    )

parser.add_argument(
                    '-a',
                    action='store',
                    type=str,
                    help='The location to save data.'
                    )


args = parser.parse_args()


def tg_plots(
             x,
             y,
             iterations,
             density,
             clusters,
             saveplots=None,
             filename=None
             ):
    '''
    The plotting code for Tg.
                                                                                                                    
    inputs:                                                                                                         
        x = The temperature data.                                                                                   
        y = The E-3kT data.                                                                                         
        iterations = The number of times to calculate Tg.                                                           
        clusters = The number of K-means clusters.                                                                  
        saveplots = Location to save plots                                                                          
                                                                                                                    
    outputs:                                                                                                        
        tg = The glass transition temperature.                                                                      
    '''                                                                                                             
                                                                                                                    
    # Sort x data ascending and remove the first couple                                                             
    minx = x[x.argsort()[:2][-1]]                                                                                   
    maxx = max(x)  # Maximum x data                                                                                 
                                                                                                                    
    rangex = np.linspace(minx, maxx, iterations)  # Grid x space                                                    
                                                                                                                    
    # Loop from x space grid and calculate various sudo Tg values                                                   
    tgs = []                                                                                                        
    xcuts = []                                                                                                      
    for i in rangex:                                                                                                
        index = (x <= i)                                                                                            
        xcut = x[index]                                                                                             
        ycut = y[index]

        xfitcut = np.linspace(min(xcut), max(xcut), density)
        yfitcut = np.interp(xfitcut, xcut, ycut)  # Linear interpolation

        tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)

        tgs.append(tg)
        xcuts.append(max(xcut))

    tgs = np.array(tgs)

    # Cluster Data
    X = np.column_stack((tgs, xcuts))
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    indexes = kmeans.labels_

    # Select cluster with minimum temperature mean
    labels = np.unique(indexes)
    tgs_cut = tgs[indexes == labels[0]]
    mean_xs = np.mean(rangex[indexes == labels[0]])
    for key in labels[1:]:

        new_mean_xs = np.mean(rangex[indexes == labels[key]])
        if new_mean_xs < mean_xs:
            mean_xs = new_mean_xs
            tgs_cut = tgs[indexes == labels[key]]

    index = np.argmax(tgs_cut)  # Choose maximum Tg
    xcut = rangex[index]

    # Truncate data based on maximum Tg from minimum cluster
    index = (x <= xcut)
    xnew = x[index]
    ynew = y[index]

    # Find Tg for filtered data
    xfitcut = np.linspace(min(xnew), max(xnew), density)
    yfitcut = np.interp(xfitcut, xnew, ynew)  # Linear interpolation

    tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)

    # Condition on plotting
    if saveplots:

        # Create plotting directory
        saveplots = os.path.join(saveplots, filename)
        if not os.path.exists(saveplots):
            os.makedirs(saveplots)


        # Plot Tg clusters
        fig, ax = pl.subplots()

        for key in labels:
            ax.plot(
                    rangex[indexes == key],
                    tgs[indexes == key],
                    marker='.',
                    linestyle='none',
                    label='Cluster '+str(key)
                    )

        ax.axvline(
                   xcut,
                   linestyle=':',
                   color='k',
                   label='Upper Temperature Cutoff: '+str(xcut)+' [K]'
                   )

        ax.set_xlabel('Upper Temperature Cutoff [K]')
        ax.set_ylabel('Tg [K]')

        ax.grid()
        ax.legend(loc='best')

        fig.tight_layout()

        fig.savefig(os.path.join(saveplots, 'tg_iteration'))

        # Plot original data with Tg
        fig, ax = pl.subplots()

        ax.plot(
                x,
                y,
                marker='.',
                linestyle='none',
                color='b',
                label='data'
                )

        ax.axvline(
                   tg,
                   linestyle='--',
                   color='g',
                   label='Tg = '+str(tg)+' [K]'
                   )

        ax.grid()
        ax.legend()

        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Potential Energy [eV/atom]')

        fig.tight_layout()

        fig.savefig(os.path.join(saveplots, 'tg'))

        # Plot error metric for cluster filtered data
        fig, ax = pl.subplots()

        ax.plot(
                endpoints,
                middle_rmse,
                marker='.',
                linestyle='none'
                )

        ax.set_xlabel('Endpoint Temperature [K]')
        ax.set_ylabel(r'RMSE $[eV^{2}]$')

        ax.grid()

        fig.tight_layout()

        fig.savefig(os.path.join(saveplots, 'rmse'))

        pl.close('all')

    return tg


def run_iterator(path, filename, tempfile, *args, **kwargs):
    '''
    Iterate through each run and gather data.

    inputs:
        path = The path where run data is.
        filename = The name of the file containing thermodynamic data.
        tempfile = The name of the file containing the hold temperature.
        k = Spline fitting parameter
        s = Spline fitting parameter

    outputs:
        df = The data from runs.
    '''

    # Gather all applicable paths
    print(filename, path)
    paths = finder(filename, path)

    kboltz = physical_constants['Boltzmann constant in eV/K'][0]

    # Gather all data and make one dataframe
    df = []
    for item in paths:
        job = os.path.commonprefix([item, path])
        job = item.strip(job)

        cols, data = data_parse(os.path.join(item, filename))
        data = pd.DataFrame(data, columns=cols)

        # Set beggining steps to zero
        data['TimeStep'] -= min(data['TimeStep'])
        data['run'] = job

        # Hold temperature
        temp = float(
                     np.loadtxt(os.path.join(item, tempfile))
                     )  # Hold temperature

        data['hold_temp'] = temp
        df.append(data)

    df = pd.concat(df)
    df['job'] = df['run'].apply(lambda x: x.split('/run/')[0])

    # The number of atoms from naming convention
    df['atoms'] = df['job'].apply(lambda x: re.split('(\d+)', x.split('_')[1]))
    df['atoms'] = df['atoms'].apply(lambda x: [i for i in x if i.isnumeric()])
    df['atoms'] = df['atoms'].apply(lambda x: [int(i) for i in x])
    df['atoms'] = df['atoms'].apply(lambda x: sum(x))

    # Calculate E-3kT
    df['etot'] = df['pe']+df['ke']
    df['etot/atoms'] = df['etot']/df['atoms']
    df['etot/atoms-3kT'] = df['etot/atoms']-3.0*kboltz*df['hold_temp']

    # Group data by run
    groups = df.groupby(['job', 'hold_temp'])
    mean = groups.mean().add_suffix('_mean').reset_index()
    groups = mean.groupby(['job'])

    runs = []
    tgs = []
    for i, j in groups:

        print(i)
        x = j['temp_mean'].values
        y = j['etot/atoms-3kT_mean'].values

        tg = tg_plots(
                      x,
                      y,
                      *args,
                      **kwargs,
                      )

        runs.append(i)
        tgs.append(tg)


    df = pd.DataFrame({'job': runs, 'tg': tgs})

    return df


df = run_iterator(args.d, args.n, args.t, args.s, args.v, args.k)

df.to_csv(os.path.join(args.a, 'tg.txt'), index=False)
