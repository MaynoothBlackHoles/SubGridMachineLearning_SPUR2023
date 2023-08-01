#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:13:38 2023

@author: john
"""

import os
import glob

import yt
# import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation


current_directory = os.getcwd()
files = glob.glob(current_directory + '/snap*.hdf5')
files = sorted(files)
datasets = [yt.load(file) for file in files]


plot = yt.SlicePlot(datasets[0], "z", ("gas", "density"))
# plot.set_zlim(("gas", "density"), 8e-29, 3e-26)

fig = plot.plots[("gas", "density")].figure

# animate must accept an integer frame number. We use the frame number
# to identify which dataset in the time series we want to load
def animate(i):
    ds = datasets[i]
    plot._switch_ds(ds)
    plot.annotate_title(f'redshift {ds.current_redshift}')
    

animation = FuncAnimation(fig, animate, frames=len(datasets))

# Override matplotlib's defaults to get a nicer looking font
with rc_context({"mathtext.fontset": "stix"}):
    animation.save(current_directory + "/animation.gif")