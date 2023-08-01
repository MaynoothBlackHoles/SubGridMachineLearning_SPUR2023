import torchvision.transforms.v2 as transforms
import torchvision
import torch
import random
import numpy as np
import scipy

import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = parent_dir.replace("\\", "/")
sys.path.append(parent_dir)

from src import subgridmodel as sgm

# parameters
IMAGE_SLICE_SIZE = 32
SCALE_FACTOR = 2 # int, float or tuple
SIZE = 100 
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

# data transforms
data_ToTensor = transforms.ToTensor()

tensors_dict = np.load("snap_007_tensors.npz")
tensors_list = []
for key in tensors_dict:
    tensors_list.append(tensors_dict[key])
print(len(tensors_list))

def transform_tensors(tensors, transform=transforms.ToTensor()):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors

def downscale_tensors(tensors, scale_factor):
    transformed_tensors = []
    scale = (1/scale_factor, 1/scale_factor, 1/scale_factor)

    for i, tensor in enumerate(tensors):
        tensor = scipy.ndimage.zoom(tensor, scale,)
        transformed_tensors.append(tensor)
    return transformed_tensors


print("[INFO] Creating datasets")
sliced_tensor_list = sgm.sr_data_slicer(tensors_list, IMAGE_SLICE_SIZE, tensor_slicer=sgm.tensor_slicer_3d)
random.shuffle(sliced_tensor_list)

print("[INFO] Transforming tensors")
downscaled = downscale_tensors(sliced_tensor_list)
high_res = transform_tensors(sliced_tensor_list)

split_num = int(len(sliced_tensor_list) * 0.9)
print(f"[INFO] Total amount of samples: {len(sliced_tensor_list)}")

training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

# saving data
print("[INFO] Saving datasets")
dictionary = {"training": training,
              "validation": validation,
              "properties": {"image patch size": IMAGE_SLICE_SIZE, "dataset size": SIZE, "scale factor": SCALE_FACTOR, "Gaussian blur": True}}

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

save_data(dictionary, name=f"dataset_{SIZE}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}", )

print("[INFO] Done!")