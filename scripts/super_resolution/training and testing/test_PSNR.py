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
sys.path.append(parent_dir)

from src import network_function as nf
from src import subgridmodel as sgm

# load your dataset
dataset_name = "dataset_100_33_[8].pt"
test_data = torch.load(current_dir + f"/data/{dataset_name}")
test_data = sgm.batch_classified_data(classified_dataset=test_data["training"], batch_size=32)

# testing on loaded dataset
data_PSNR = nf.test_PSNR(test_data)

print("-----------------------------")
print(data_PSNR)

"""
for scalefactor of 2: PSNR = 29.3
for scalefactor of 3: PSNR = 27.3
for scalefactor of 4: PSNR = 25.2
for scalefactor of 8: PSNR = 8.84

"""