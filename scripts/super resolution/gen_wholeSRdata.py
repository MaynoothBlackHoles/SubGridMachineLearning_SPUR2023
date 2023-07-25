"""
This script is for generating a sample dataset which is to be used to test the network visually, with the test_image.py script
"""

import torchvision.transforms.v2 as transforms
import torchvision
import torch
from PIL import Image
import random

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/") 
sys.path.append(parent_dir)

# parameters
IMAGE_SIZE = 500
SCALE_FACTOR = 2
LOW_RES = IMAGE_SIZE // SCALE_FACTOR
EXTRACT_SIZE = 100
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

# data transforms
data_ToTensor = transforms.ToTensor()
data_downscale = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.Resize(LOW_RES, interpolation=INTERPOLATION),
    transforms.Resize(IMAGE_SIZE, interpolation=INTERPOLATION),
    transforms.ToTensor()
])

# loading this dataset with torchvision will install, among other things, a folder with all of the images in the dataset which we take out manually with 
# the extract tensors function later on
print("[INFO] Loading dummy_data")
dummy_data = torchvision.datasets.Flowers102(root=current_dir + "/data", download=True, transform=transforms.ToTensor())

def extract_tensors(folder_location, max_size=-1):
    tensor_list = []
    tick = 0
    for image in os.listdir(folder_location):
        img = Image.open(folder_location + f"/{image}")
        tensor = data_ToTensor(img)
        tensor_list.append(tensor)

        tick += 1
        if tick == max_size:
            break

    return tensor_list

def transform_tensors(tensors, transform=transforms.ToTensor()):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors

# taking out images from the datset and converting them into torch tensors
print("[INFO] Extracting images")
tensor_list = extract_tensors(folder_location= current_dir + "/data/flowers-102/jpg", max_size=EXTRACT_SIZE)

# saving and sorting datasets
print("[INFO] Creating datasets")
print(f"[INFO] Total amount of samples: {len(tensor_list)}")
random.shuffle(tensor_list)
downscaled = transform_tensors(tensor_list, data_downscale)
high_res = transform_tensors(tensor_list)

data = (downscaled, high_res)

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

# saving data
print("[INFO] Saving datasets")
save_data(data, name=f"sample_{IMAGE_SIZE}s")

print("[INFO] Done!")