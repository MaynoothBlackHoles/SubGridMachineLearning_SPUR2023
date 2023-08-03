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
CHANNEL_NUM = 0
SCALE_NUM = 1e+28
CUBE_LEN = 120 # max is 256

CHANNEL_NUM = 0
Z_SLICE = 0

# pick names of weights and dataset
weights_name = f"rcnn3d_1_32_{SCALE_FACTOR}.pt"
dataset_name = "snap_007_tensors.npz"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.CNN_3D(depth=3, channels=1, kernel_front=9).to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
data = np.load(DATA_DIR + f"/datasets/{dataset_name}")
sample = data["region 0, 0, 200"] # shape (256,256,256,6)
sample = torch.tensor(sample).permute(3, 0, 1, 2)
sample = torch.stack([sample[CHANNEL_NUM,:CUBE_LEN,:CUBE_LEN,:CUBE_LEN] * SCALE_NUM])

downscaled_img = du.downscale_tensors([sample], SCALE_FACTOR)
init_img = model.init_conv(downscaled_img)
mid_img = model.mid_conv(init_img)
end_img = model.end_conv(mid_img)

img_list = [sample, downscaled_img, init_img, mid_img, end_img]

fig = plt.figure()
for i, image in enumerate(img_list):
    fig.add_subplot(1, len(img_list), i+1)
    image = torch.squeeze(image)
    print(image.shape)
    if len(image.shape) == 4:
        image = image[CHANNEL_NUM][Z_SLICE]
    else:
        image = image[Z_SLICE]    
    plt.imshow(image.detach())
    plt.axis("off")

plt.show()
