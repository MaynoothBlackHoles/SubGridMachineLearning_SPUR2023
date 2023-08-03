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
from subgrid_physics_modelling import data_utils as du


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pick names of weights and dataset
weights_name = "rcnn3d_32_2.pt"
dataset_name = "snap_007_tensors.npz"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.Residual_CNN_3D(channels=1, kernel_front=9).to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
data = np.load(DATA_DIR + f"/datasets/{dataset_name}")
sample = data["region 0, 0, 300"] # shape (256,256,256,6)
sample = torch.tensor(sample).permute(3, 0, 1, 2)
cube_len = 80
sample = sample[:,:cube_len,:cube_len,:cube_len]

low_res_tensor = du.downscale_tensors([sample], 8)[0][5]
high_res_tensor = sample[5]
srcnn_tensor = model(torch.stack([low_res_tensor]))

img_list = [low_res_tensor, high_res_tensor, srcnn_tensor]

import matplotlib.pyplot as plt

fig = plt.figure()
for i, image in enumerate(img_list):
    fig.add_subplot(1, 3, i+1)
    image = torch.squeeze(image)
    image = image[0]
    plt.imshow(image.detach())
    plt.axis("off")

plt.show()