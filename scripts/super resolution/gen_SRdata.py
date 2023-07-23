from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import torchvision
import torch

import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/")

IMAGE_SIZE = 500
LOW_RES = int(IMAGE_SIZE/3)

RE_SCALED_SIZE = IMAGE_SIZE #+ 12
INTERPOLATION = torchvision.transforms.InterpolationMode.BICUBIC

data_transform = transforms.Compose([
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

trainData = torchvision.datasets.Flowers102(root="data", split="train", download=True, transform=data_transform)
downscaled_trainData = torchvision.datasets.Flowers102(root="data", split="train", download=True, transform=data_downscale)
trainDataLoader = DataLoader(trainData, )
downscaled_trainDataLoader = DataLoader(downscaled_trainData)

"""valData = torchvision.datasets.Flowers102(root="data", split="test", download=True, transform=data_transform)
downscaled_valData = torchvision.datasets.Flowers102(root="data", split="val", download=True, transform=data_downscale)
valDataLoader = DataLoader(valData)
downscaled_valDataLoader = DataLoader(downscaled_valData)"""

def extract_images(dataLoader, img_size):
    tensors = []
    for i, (x, y) in enumerate(dataLoader):
        x = torch.reshape(x, (3, img_size, img_size))
        tensors.append(x)
    return tensors

tensor_trainData = extract_images(trainDataLoader, IMAGE_SIZE)
tensor_downscaled_trainData = extract_images(downscaled_trainDataLoader, RE_SCALED_SIZE)
#tensor_valData = extract_images(valDataLoader, IMAGE_SIZE)
#tensor_downscaled_ValData = extract_images(downscaled_trainData, RE_SCALED_SIZE)

def save_data(dataset, name):
    torch.save(dataset, current_dir + f"/data/{name}.pt")

size = len(tensor_trainData)
print(size)
split_num = int(size * 0.9)

training = (tensor_downscaled_trainData[:split_num], tensor_trainData[:split_num])
validation = (tensor_downscaled_trainData[split_num:], tensor_trainData[split_num:])

save_data(training, name=f"training_{IMAGE_SIZE}s")
save_data(validation, name=f"validation_{IMAGE_SIZE}s")
    