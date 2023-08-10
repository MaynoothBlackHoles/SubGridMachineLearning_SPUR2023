"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch
import numpy as np

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution"

from subgrid_physics_modelling import super_resolution_networks as net

# pick names of weights and dataset
weights_name = "rcnn3d_125_32_8.pt"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.CNN_3D(channels=1, kernel_front=9, mid_channels=16)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

layers = [p for p in model.parameters()]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

n, m = 4, 4 
shift = n*m * 0
index = np.arange(shift, n*m + shift).reshape((n, m))
fig = make_subplots(rows=n, cols=m, specs=[ [{"type": "volume"}] * m] * n)

for i in range(n):
    for j in range(m):

        volume_matrix = layers[0][index[i,j]][0].detach().numpy() * 20 # this product makes to plots clearer to see

        step_num_x = complex(0,len(volume_matrix))
        step_num_y = complex(0,len(volume_matrix[0]))
        step_num_z = complex(0,len(volume_matrix[0][0]))
        X, Y, Z = np.mgrid[0:1:step_num_x, 0:1:step_num_y, 0:1:step_num_z]

        fig.add_trace(
            go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=volume_matrix.flatten(),
                isomin=0,
                isomax=1,
                opacity=0.1, # needs to be small to see through all surfaces
                surface_count=10, # needs to be a large number for good volume rendering
                ),
            row=i+1, col=j+1)

fig.show()