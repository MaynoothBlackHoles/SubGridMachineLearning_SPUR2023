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

# parameters
IMAGE_SLICE_SIZE = 33
MAX_EXTRACT = -1
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

# data transforms
data_ToTensor = transforms.ToTensor()
data_downscale_sf2 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.Resize(IMAGE_SLICE_SIZE //2, interpolation=INTERPOLATION),
    transforms.Resize(IMAGE_SLICE_SIZE, interpolation=INTERPOLATION),
    transforms.ToTensor()
])
data_downscale_sf3 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.Resize(IMAGE_SLICE_SIZE//3, interpolation=INTERPOLATION),
    transforms.Resize(IMAGE_SLICE_SIZE, interpolation=INTERPOLATION),
    transforms.ToTensor()
])
data_downscale_sf4 = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.Resize(IMAGE_SLICE_SIZE//4, interpolation=INTERPOLATION),
    transforms.Resize(IMAGE_SLICE_SIZE, interpolation=INTERPOLATION),
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
tensor_list = extract_tensors(folder_location= current_dir + "/data/flowers-102/jpg", max_size=MAX_EXTRACT)

# saving and sorting datasets
print("[INFO] Creating datasets")
sliced_tensor_list = sgm.sr_data_slicer(tensor_list, IMAGE_SLICE_SIZE)
random.shuffle(sliced_tensor_list)
print(f"[INFO] Total amount of samples: {len(sliced_tensor_list)}")
downscaled_sf2 = transform_tensors(sliced_tensor_list, data_downscale_sf2)
downscaled_sf2 = torch.stack(downscaled_sf2)

downscaled_sf3 = transform_tensors(sliced_tensor_list, data_downscale_sf3)
downscaled_sf3 = torch.stack(downscaled_sf3)

downscaled_sf4 = transform_tensors(sliced_tensor_list, data_downscale_sf4)
downscaled_sf4 = torch.stack(downscaled_sf4)

downscaled = torch.stack((downscaled_sf2, downscaled_sf3, downscaled_sf4))

high_res = transform_tensors(sliced_tensor_list)
high_res = torch.stack((high_res, high_res, high_res))

split_num = int(len(sliced_tensor_list) * 0.9) * 3
training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

# saving data
print("[INFO] Saving datasets")
save_data(training, name=f"training_sf234_gb_{IMAGE_SLICE_SIZE}ss")
save_data(validation, name=f"validation_sf234_gb_{IMAGE_SLICE_SIZE}ss")

print("[INFO] Done!")