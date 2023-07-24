"""
Script to compare images to their bicubic inertpolation and after run through trained model
"""

import torch
from torch import nn
import torchvision

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/")
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/")
sys.path.append(parent_dir)

from src import sr_networks as net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_name = "srcnn33_slices.pt"
dataset_name = "training_500s.pt"

model = net.Srcnn().to(device)
model.load_state_dict(torch.load(current_dir + f"/network weights/{weights_name}"))
loss_fn = nn.MSELoss()

print("[INFO] Loading datasets")
test_data = torch.load(current_dir + f"/data/{dataset_name}")

sample_num = 567

low_res_tensor = test_data[0][sample_num]
high_res_tensor = test_data[1][sample_num]
srcnn_tensor = model(low_res_tensor)

transform = torchvision.transforms.ToPILImage()

img = transform(high_res_tensor)
img.show()
img = transform(low_res_tensor)
img.show()
img = transform(srcnn_tensor)
img.show()

print("[INFO] Done! :D")