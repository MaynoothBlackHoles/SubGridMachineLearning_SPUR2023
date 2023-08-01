"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/") 
sys.path.append(parent_dir)

from src import sr_networks as net

# pick names of weights and dataset
weights_name = "srcnn33_slices_blured.pt"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.Srcnn()
model.load_state_dict(torch.load(current_dir + f"/network weights/{weights_name}"))

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
