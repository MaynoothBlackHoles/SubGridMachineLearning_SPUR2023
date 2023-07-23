import torchvision.transforms.v2 as transforms
import torchvision
import torch
from PIL import Image
import random

import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/")
sys.path.append(parent_dir)

from src import subgridmodel as sgm

ORIGINAL_IMAGE_CROP = 500
IMAGE_SLICE_SIZE = 50
LOW_RES = int(IMAGE_SLICE_SIZE/3)

RE_SCALED_SIZE = IMAGE_SLICE_SIZE #+ 12
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

data_cropToTensor = transforms.Compose([
	transforms.CenterCrop(ORIGINAL_IMAGE_CROP),
    transforms.ToTensor()
])

data_downscale = transforms.Compose([
	transforms.CenterCrop(IMAGE_SLICE_SIZE),
    #transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2)),
    transforms.Resize(LOW_RES, interpolation=INTERPOLATION),
    transforms.Resize(RE_SCALED_SIZE, interpolation=INTERPOLATION),
    transforms.ToTensor()
])

print("[INFO] Loading dummy_data")
dummy_data = torchvision.datasets.Flowers102(root=current_dir + "/data", download=True, transform=transforms.ToTensor())

print("[INFO] Extracting images")
tensor_list = []
#MAX_LIST_SIZE = 300
#tick = 0
for image in os.listdir(current_dir + "/data/flowers-102/jpg"):
    #tick += 1
    #if tick == MAX_LIST_SIZE:
    #    break
    img = Image.open(current_dir + f"/data/flowers-102/jpg/{image}")
    tensor = data_cropToTensor(img)
    tensor_list.append(tensor)

tensor_list =  torch.stack(tensor_list)
print(tensor_list.shape)

def transform_tensors(tensors, transform=transforms.ToTensor()):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors

print("[INFO] Creating datasets")
sliced_tensor_list = sgm.sr_data_slicer(tensor_list, IMAGE_SLICE_SIZE)
random.shuffle(sliced_tensor_list)
downscaled = transform_tensors(sliced_tensor_list, data_downscale)
high_res = transform_tensors(sliced_tensor_list)


split_num = int(len(sliced_tensor_list) * 0.9)
print(f"[INFO] Total amount of samples: {len(sliced_tensor_list)}")

training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

print("[INFO] Saving datasets")
save_data(training, name=f"sliced_training_{IMAGE_SLICE_SIZE}s")
save_data(validation, name=f"sliced_validation_{IMAGE_SLICE_SIZE}s")

print("[INFO] Done!")