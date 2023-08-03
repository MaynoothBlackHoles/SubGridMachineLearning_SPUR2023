"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution"

from subgrid_physics_modelling import super_resolution_networks as net

# pick names of weights and dataset
weights_name = "srcnn33_slices_blured.pt"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.SRcnn()
model.load_state_dict(torch.load(DATA_DIR + f"/network weights/{weights_name}"))

layers = [p for p in model.parameters()]
print(layers)

import matplotlib.pyplot as plt

fig = plt.figure()
columns = 8
rows = 8
for i in range(1, columns*rows +1):
    filter = layers[0][i - 1][1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(filter.detach())
    plt.axis("off")

plt.colorbar()
plt.show()
