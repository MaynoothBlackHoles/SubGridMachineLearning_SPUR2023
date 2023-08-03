"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch
import matplotlib.pyplot as plt
import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution"

from subgrid_physics_modelling import super_resolution_networks as net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pick names of weights and dataset
weights_name = "srcnn33_slices_blured.pt"
dataset_name = "sample_2.pt"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.SRcnn().to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network weights/{weights_name}"))

print("[INFO] Loading datasets")
test_data = torch.load(DATA_DIR + f"/datasets/{dataset_name}")
test_data = test_data["images"]
# position of image in dataset
sample_num = 90

# extracting tensors
low_res_tensor = test_data[0][sample_num]
high_res_tensor = test_data[1][sample_num]
srcnn_tensor = model(low_res_tensor)

img_list = [low_res_tensor, high_res_tensor, srcnn_tensor]


fig = plt.figure()
for i, image in enumerate(img_list):
    fig.add_subplot(1, 3, i+1)
    plt.imshow(image.detach())
    plt.axis("off")

plt.show()