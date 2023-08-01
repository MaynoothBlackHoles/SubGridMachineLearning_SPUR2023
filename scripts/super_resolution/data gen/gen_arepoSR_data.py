import torchvision.transforms.v2 as transforms
import torchvision
import torch
import random
import numpy as np

import os
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
top_dir = top_dir.replace("\\", "/")
os.chdir(top_dir)

from subgrid_physics_modelling import data_utils as du 

# parameters
IMAGE_SLICE_SIZE = 32
SCALE_FACTOR = 2 # int, float or tuple

DATA_DIR = top_dir + "/data/super_resolution/datsets"

tensors_dict = np.load(DATA_DIR + "/snap_007_tensors.npz")
tensors_list = []
for key in tensors_dict:
    tensors_list.append(tensors_dict[key])

size = len(tensors_list)

print("[INFO] Creating datasets")
sliced_tensor_list = du.sr_data_slicer(tensors_list, IMAGE_SLICE_SIZE, tensor_slicer=du.tensor_slicer_3d)
random.shuffle(sliced_tensor_list)

print("[INFO] Transforming tensors")
downscaled = du.downscale_tensors(sliced_tensor_list, scale_factor=SCALE_FACTOR)
high_res = du.transform_tensors(sliced_tensor_list)

split_num = int(len(sliced_tensor_list) * 0.9)
print(f"[INFO] Total amount of samples: {len(sliced_tensor_list)}")

training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

# saving data
print("[INFO] Saving datasets")
data_dict = {"training": training,
              "validation": validation,
              "properties": {"image patch size": IMAGE_SLICE_SIZE, "dataset size": size, "scale factor": SCALE_FACTOR}}

def save_data(dataset, name):
    torch.save(dataset, DATA_DIR + f"/{name}.pt")

save_data(data_dict, name=f"dataset_{size}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}", )

print("[INFO] Done!")