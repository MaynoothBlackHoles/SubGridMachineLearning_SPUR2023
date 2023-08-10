"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch
import matplotlib.pyplot as plt
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
from subgrid_physics_modelling import network_training_utils as ntu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pick names of weights and dataset
weights_name = "srcnn33_slices_blured.pt"
dataset_name = "sample_2.pt"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.SRcnn().to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
test_data = torch.load(DATA_DIR + f"/datasets/{dataset_name}")
test_data = test_data["images"]
# position of image in dataset
sample_num = 54

# extracting tensors
high_res_tensor = test_data[1][sample_num]
low_res_tensor = test_data[0][sample_num]
srcnn_tensor = model(low_res_tensor)

img_list = [high_res_tensor, low_res_tensor ,srcnn_tensor]
label_list = ["input", "bicubic", "SRCNN"]

fig = plt.figure()
for i, image in enumerate(img_list):
    fig.add_subplot(1, 3, i+1)
    image = image.permute(1, 2, 0).detach()
    plt.imshow(image)
    plt.title(f"{label_list[i]}")
    plt.axis("off")

plt.show()