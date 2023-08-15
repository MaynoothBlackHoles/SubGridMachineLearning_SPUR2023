#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:21:25 2023

@author: john
"""

import os
import numpy as np
import torchvision
import torch

from scipy.ndimage import gaussian_filter, zoom

from srcnn import SRCNN
import srcnn

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import plotly


#%%
current_directory = os.getcwd()
data_path = current_directory + '/../../data/tensors128.npz'

tensors = np.load(data_path)

#%%
Xs = [X for X in tensors.values()]

#%%
import plotly.graph_objects as go

N = 64
X, Y, Z = np.mgrid[0:N, 0:N, 0:N]
values = np.log10(Xs[-1][0:N, 0:N, 0:N])

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=2.0,
    isomax=3,
    opacity=0.05,      # needs to be small to see through all surfaces
    surface_count=40, # needs to be a large number for good volume rendering
    ))

#%%
from plotly.offline import plot
plot(fig, auto_open=True)

#%%
plt.imshow(np.log10(np.sum(Xs[-1], axis=2)))