"""
Script to test avarage PSNR of a chosen dataset
"""

import torch

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/") 

from src import network_function as nf

# load your dataset
dataset_name = input("Dataset name: ")
train_data = torch.load(current_dir + f"/data/.pt")

# testing on loaded dataset
data_PSNR = nf.test_PSNR(train_data)

print("-----------------------------")
print(data_PSNR) #32.97324611813414