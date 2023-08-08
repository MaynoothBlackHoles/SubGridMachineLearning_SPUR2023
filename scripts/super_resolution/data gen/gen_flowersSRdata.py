import torchvision.transforms.v2 as transforms
import torchvision
import torch
from PIL import Image
import random

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution/datasets"

from subgrid_physics_modelling import data_utils as du

# parameters
IMAGE_SLICE_SIZE = 33
SCALE_FACTORS = 8 # int, float or tuple
SIZE = 100 # size of datset will be the chosen number multiplied by the amount of tuples specifed in line above, if you want max exements slot in -1
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

# data transforms
data_ToTensor = transforms.ToTensor()

# loading this dataset with torchvision will install, among other things, a folder with all of the images in the dataset which we take out manually with 
# the extract tensors function later on
print("[INFO] Loading dummy_data")
dummy_data = torchvision.datasets.Flowers102(root=DATA_DIR, download=True, transform=transforms.ToTensor())

def extract_tensors(folder_location):
    tensor_list = []
    for image in os.listdir(folder_location):
        img = Image.open(folder_location + f"/{image}")
        tensor = data_ToTensor(img)
        tensor_list.append(tensor)

    return tensor_list

def transform_tensors(tensors, transform=transforms.ToTensor()):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors

# taking out images from the datset and converting them into torch tensors
print("[INFO] Extracting images")
tensor_list = extract_tensors(folder_location=DATA_DIR + "/flowers-102/jpg")
random.shuffle(tensor_list)
tensor_list = tensor_list[:SIZE]

# saving and sorting datasets
print("[INFO] Creating datasets")
sliced_tensor_list = du.sr_data_slicer(tensor_list, IMAGE_SLICE_SIZE)

downscaled = []
high_res = []
single_sf_datasets = []

print("[INFO] Looping though scale factors")
original_data = transform_tensors(sliced_tensor_list)

if type(SCALE_FACTORS) == int:
    iterable = [SCALE_FACTORS]
for scale_factor in iterable:
    random.shuffle(sliced_tensor_list)
    data_downscale = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 5)),
        transforms.Resize(IMAGE_SLICE_SIZE // scale_factor, interpolation=INTERPOLATION),
        transforms.Resize(IMAGE_SLICE_SIZE, interpolation=INTERPOLATION),
        transforms.ToTensor()])
    
    scaled_data = transform_tensors(sliced_tensor_list, data_downscale)

    high_res.extend(original_data)
    downscaled.extend(scaled_data)

    single_sf_datasets.append([scaled_data,high_res])

# fancy code to shuffle the datasets nicely
c = list(zip(downscaled, high_res))
random.shuffle(c)
downscaled, high_res = zip(*c)

downscaled = torch.stack(downscaled)
high_res = torch.stack(high_res)

# creating the training/validation split
split_num = int(len(sliced_tensor_list) * 0.9)
training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

# saving data
print("[INFO] Saving datasets")
dictionary = {"training": training, 
              "validation": validation,
              "single sf data": single_sf_datasets,
              "properties": {"image patch size": IMAGE_SLICE_SIZE, "dataset size": SIZE, "scale factor": SCALE_FACTORS, "Gaussian blur": True}}

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

save_data(dictionary, name=f"dataset_{SIZE}_{IMAGE_SLICE_SIZE}_{SCALE_FACTORS}", )

print("[INFO] Done!")