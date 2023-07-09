#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:09:36 2023

@author: john
"""
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

import yt
from yt.units import dimensions
from yt.units.physical_constants import gravitational_constant_cgs as G
from yt.units.physical_constants import hydrogen_mass



#%%
def _H2_fraction(field, data):
    return data[('PartType0', 'ChemicalAbundances')][:, 0]


yt.add_field(
    ("PartType0", "H2_fraction"),
    function=_H2_fraction,
    sampling_type="local",
    units="auto",
    dimensions=dimensions.dimensionless,
)

N = 64
if len(sys.argv) > 1:
    N = sys.argv[1]

current_directory = os.getcwd()
files = glob.glob(current_directory + '/snap*.hdf5')
datasets = [yt.load(file) for file in files]
grids = [ds.arbitrary_grid([0, 0, 0], [100, 100, 100], dims=[N, N, N]) 
         for ds in datasets]



#%%
num_densities = []
for grid in grids:
    n = (grid[('density')] / hydrogen_mass).value
    num_densities.append(n.flatten())
    
    
    
#%%
plots_dir = current_directory + '/plots'
density_dir = '/number_density'

if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

if not os.path.exists(plots_dir + density_dir):
    os.mkdir(plots_dir + density_dir)
    
filename = plots_dir + density_dir + '/number_density_{:.3f}.png'

for i, n in enumerate(num_densities):
    plt.figure()
    redshift = datasets[i].current_redshift
    plt.hist(n, bins=50)
    plt.xlim((7e-4, 1e2))
    plt.ylim((5e-1, 2e5))
    
    plt.title(f'redshift {redshift:.3f}')
    plt.xlabel('number desnity')
    plt.ylabel('count')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename.format(redshift), dpi=300)
   
    
   
#%%
H2_fractions = []
for grid in grids:
    n = grid[('PartType0', 'H2_fraction')].value
    H2_fractions.append(n.flatten())
    
    
    
#%%
H2_dir = '/H2_fraction'
if not os.path.exists(plots_dir + H2_dir):
    os.mkdir(plots_dir + H2_dir)
    
filename = plots_dir + H2_dir + '/H2_fraction_{:.3f}.png'

for i, h2 in enumerate(H2_fractions):
    plt.figure()
    redshift = datasets[i].current_redshift
    plt.hist(h2, bins=50)
    plt.xlim((1e-6, 1e-3))
    plt.ylim((5e-1, 2e5))
    
    plt.title(f'redshift {redshift:.3f}')
    plt.xlabel('H2 fraction')
    plt.ylabel('count')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename.format(redshift), dpi=300)
    
    
    
#%%
cooling_times = []
dynamic_times = []
for ds, grid in zip(datasets, grids):
    ct = grid[('CoolTime')].value * ds.time_unit.to('s').value
    dt = np.sqrt(3 * np.pi / (32 * G * grid[('density')])).value
    cooling_times.append(ct.flatten())
    dynamic_times.append(dt.flatten())
    
    
    
#%%
time_scales_dir = '/time_scales'
if not os.path.exists(plots_dir + time_scales_dir):
    os.mkdir(plots_dir + time_scales_dir)
    
filename = plots_dir + time_scales_dir + '/time_scales_{:.3f}.png'

for (i, (ct, dt)) in enumerate(zip(cooling_times, dynamic_times)):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    redshift = datasets[i].current_redshift
    
    ax1.hist(ct, bins=100, range=(1e14, 1e18))
    ax1.set_xlabel('Cooling time (seconds)')
    ax1.set_ylabel('count')
    ax1.set_yscale('log')
    
    ax2.hist(dt, bins=100, range=(1e14, 1e18))
    ax2.set_xlabel('Free fall time (seconds)')
    ax2.set_ylabel('count')
    ax2.set_yscale('log')
    
    fig.suptitle(f'redshift {redshift:.3f}')
    plt.xscale('log')
    plt.savefig(filename.format(redshift), dpi=300)