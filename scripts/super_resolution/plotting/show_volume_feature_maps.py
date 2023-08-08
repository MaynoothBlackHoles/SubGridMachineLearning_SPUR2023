"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution"

from subgrid_physics_modelling import super_resolution_networks as net
from subgrid_physics_modelling import data_utils as du

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE_FACTOR = 8
SCALE_NUM = 1e+28
TENSOR_CHANNEL = 0
CUBE_LEN = 256 # max is 256

CHANNEL_NUM = 5
Z_SLICE = 0

# pick names of weights and dataset
weights_name = f"rcnn3d_50_32_{SCALE_FACTOR}.pt"
dataset_name = "snap_007_tensors.npz"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.CNN_3D(depth=3, channels=1, kernel_front=9, mid_channels=64).to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
data = np.load(DATA_DIR + f"/datasets/{dataset_name}")
sample = data["region 0, 0, 100"] # shape (256,256,256,6)
sample = torch.tensor(sample).permute(3, 0, 1, 2)
sample = torch.stack([sample[TENSOR_CHANNEL, :CUBE_LEN, :CUBE_LEN, :CUBE_LEN] * SCALE_NUM])

downscaled_img = du.rescale_tensors([sample], SCALE_FACTOR)
init_img = model.init_conv(downscaled_img)
mid_img = model.mid_conv(init_img)
end_img = model.end_conv(mid_img)

#volume_list = [sample, downscaled_img, init_img, mid_img, end_img]
volume_list = [sample]#, downscaled_img, init_img, mid_img, end_img]

print(sample)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

n, m = 1, 1
shift = n*m * 0
fig = make_subplots(rows=n, cols=m, specs=[ [{"type": "volume"}] * m] * n)
step = 2**5

for i, volume in enumerate(volume_list):
    volume_matrix = volume.detach().numpy()[::step, ::step, ::step]
    volume_matrix = volume_matrix / np.linalg.norm(volume_matrix) * 10


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
        row=1, col=i+1)

fig.show()