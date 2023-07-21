import torchvision.transforms.v2 as transforms
import torchvision
import torch
from PIL import Image

import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


IMAGE_SIZE = 500
LOW_RES = int(IMAGE_SIZE/3)

RE_SCALED_SIZE = IMAGE_SIZE + 12
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

data_cropToTensor = transforms.Compose([
	transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

data_downscale = transforms.Compose([
	transforms.CenterCrop(IMAGE_SIZE),
    #transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2)),
    transforms.Resize(LOW_RES, interpolation=INTERPOLATION),
    transforms.Resize(RE_SCALED_SIZE, interpolation=INTERPOLATION),
    transforms.ToTensor()
])

convert_tensor = transforms.ToTensor()

print("Loading dummy_data")
dummy_data = torchvision.datasets.Flowers102(root=current_dir + "/data", download=True, transform=transforms.ToTensor())

print("")
tensor_list = []
for image in os.listdir(current_dir + "/data/flowers-102/jpg"):
    img = Image.open(current_dir + f"/data/flowers-102/jpg/{image}")
    tensor = convert_tensor(img)
    tensor_list.append(tensor)

def transform_tensors(tensors, transform):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors

downscaled = transform_tensors(tensor_list, data_downscale)
high_res = transform_tensors(tensor_list, data_cropToTensor)

split_num = int(len(tensor_list) * 0.9)
print(len(tensor_list))

training = downscaled[:split_num], high_res[:split_num] 
validation = downscaled[split_num:], high_res[split_num:] 

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

save_data(training, name=f"training_{IMAGE_SIZE}s")
save_data(validation, name=f"validation_{IMAGE_SIZE}s")