#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:39:49 2023

@author: john
"""

import os
import glob
import yt

current_directory = os.getcwd()
files = glob.glob(current_directory + '/snap*.hdf5')
files = sorted(files)
datasets = [yt.load(file) for file in files]

plots_dir = current_directory + '/plots'
slice_dir = '/slice_plots'

if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

if not os.path.exists(plots_dir + slice_dir):
    os.mkdir(plots_dir + slice_dir)
    
filename = plots_dir + slice_dir + '/slice_redshift_{:.3f}.png'

for ds in datasets:
    plot = yt.SlicePlot(ds, "z", ("gas", "density"))
    plot.annotate_title(f'redshift {ds.current_redshift}')
    plot.save(filename.format(ds.current_redshift))