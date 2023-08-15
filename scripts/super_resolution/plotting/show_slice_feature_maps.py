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
from subgrid_physics_modelling import network_training_utils as ntu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE_FACTOR = 4
SCALE_NUM = 1#1e+28
TENSOR_CHANNEL = 0 
CUBE_LEN = 128 # max is 128 # keep the scalefactor as a multiple of this number to get matching size for rescaled image
CHANNEL_NUM = 2
Z_SLICE = 0

# pick names of weights and dataset
weights_name = f"cnn3d_16_32_4.pt"
dataset_name = "tensors128.npz"

# loading network architecture and saved weights
print("[INFO] Loading network")
model = net.CNN_3D(mid_channels= 64, kernel_front=9, kernel_mid=3, kernel_end=3).to(device)
model.load_state_dict(torch.load(DATA_DIR + f"/network_weights/{weights_name}"))

print("[INFO] Loading datasets")
data = np.load(DATA_DIR + f"/datasets/{dataset_name}")

for key in data:
    print(key)

num = "100"
sample = data[f"ENZOE-000{num}/ENZOE-000{num}.block_list"]
sample = torch.tensor(sample)
sample = torch.stack([sample[:CUBE_LEN, :CUBE_LEN, :CUBE_LEN] * SCALE_NUM])

downscaled_img = du.rescale_tensors([sample], SCALE_FACTOR)
#init_img = model.init_conv(downscaled_img.float())
#mid_img = model.mid_conv(init_img)
#end_img = model.end_conv(mid_img)

y = model(downscaled_img.float())

#diff1 = torch.abs(sample - downscaled_img)
#diff2 = torch.abs(sample - end_img)

#feature_maps = [sample, downscaled_img, init_img, mid_img, end_img]
feature_maps = [sample, downscaled_img, y]
#differences = [diff1, diff2]

def plot_figs(name, tensors_list):
    fig = plt.figure()

    for i, image in enumerate(tensors_list):
        fig.add_subplot(1, len(tensors_list), i+1)
        image = torch.squeeze(image)

        
        if len(image.shape) == 4:
            image = image[CHANNEL_NUM,Z_SLICE]
        else:
            image = image[Z_SLICE]  
        
        psnr = round(float(ntu.eval_MSE(sample[0][Z_SLICE], image)), 2)
        
        plt.imshow(image.detach())
        plt.axis("off")
        plt.title(f"{psnr}")
        plt.colorbar()

    plt.savefig(DATA_DIR + f"/plots/{name}")
    plt.show()

plot_figs("fmaps", feature_maps)
#plot_figs("diff", differences)