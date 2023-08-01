#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:05:09 2023

@author: john

This script will read the snap_*.hdf5 files dumped by arepo in the current
working directory and write numpy tensors with field data extracted from the
snapshots to a file.
"""

import os
import sys
import glob

import yt
import numpy as np
import yt.units.physical_constants.gravitational_constant_cgs as G
import yt.units.physical_constants.hydrogen_mass as hydrogen_mass

from yt.units import dimensions


def _H2_fraction(field, data):
    return data[('PartType0', 'ChemicalAbundances')][:, 0]

yt.add_field(
    ("PartType0", "H2_fraction"),
    function=_H2_fraction,
    sampling_type="local",
    units="auto",
    dimensions=dimensions.dimensionless,
)

current_directory = os.getcwd()
files = glob.glob(current_directory + 'snap*.hdf5')
datasets = [yt.load(file) for file in files]


N = 64
if len(sys.argv) > 1:
    N = sys.argv[1]
tensors = {}


for file ,ds in zip(files, datasets):
    obj = ds.arbitrary_grid([0, 0, 0], [100, 100, 100], dims=[N, N, N])
    X = np.zeros((N, N, N, 9))
    
    # Number density
    X[:, :, :, 0] = (obj[('density')] / hydrogen_mass).value
    
    # H2 density
    X[:, :, :, 1] = obj[('PartType0', 'H2_fraction')].value
    
    # Velocity and divergence
    X[:, :, :, 2] = obj[('velocity_x')].value
    X[:, :, :, 3] = obj[('velocity_y')].value
    X[:, :, :, 4] = obj[('velocity_z')].value
    X[:, :, :, 5] = obj[('VelocityDivergence')].value
    
    # Temperature and cooling time
    X[:, :, :, 6] = obj[('temperature')].value
    X[:, :, :, 7] = (obj[('CoolTime')] * ds.time_unit.to('s')).value
    
    # Dynamical time
    X[:, :, :, 7] = np.sqrt(3 * np.pi / (32 * G * obj[('density')])).value
    
    tensors[file] = X

np.savez(current_directory + 'tensors.npz', **tensors)
# loaded_data = np.load('tensors.npz')