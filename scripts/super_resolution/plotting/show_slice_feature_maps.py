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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pick names of weights and dataset
weights_name = "rcnn3d_32_2.pt"
dataset_name = "snap_007_tensors.npz"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.Residual_CNN_3D(depth=3, channels=1, kernel_front=9).to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
data = np.load(DATA_DIR + f"/datasets/{dataset_name}")
sample = data["region 0, 0, 300"] # shape (256,256,256,6)
sample = torch.tensor(sample).permute(3, 0, 1, 2)
cube_len = 120
sample = torch.stack([sample[:,:cube_len,:cube_len,:cube_len][5]])

layers = [p for p in model.parameters()] 
#[torch.Size([64, 1, 9, 9, 9]), torch.Size([64]), torch.Size([64, 64, 3, 3, 3]), torch.Size([64]), torch.Size([1, 64, 3, 3, 3]), torch.Size([1])]

init_img = model.init_conv(sample)
mid_img = model.mid_conv(init_img)
residue_img = model.end_conv(mid_img)
final_img = residue_img + sample

img_list = [sample, init_img, mid_img, residue_img, final_img]

fig = plt.figure()
for i, image in enumerate(img_list):
    fig.add_subplot(1, len(img_list), i+1)
    image = torch.squeeze(image)
    print(image.shape)
    if len(image.shape) == 4:
        image = image[0][0]
    else:
        image = image[0]    
    plt.imshow(image.detach())
    plt.axis("off")

plt.show()
