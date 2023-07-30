"""
Script with an arbitrary input size of data which uses a network trained on smaller sized data
"""

import torch
from torch import nn
import time
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm
from src import network_function as nf
from src import networks as net

BL = 64 # size of input data
MODEL_SIZE = 16 # size of data which chosen trained newtork was trained on

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Kernel1_conv(box_length=MODEL_SIZE).to(device)
# loading networks weights from specified location
model.load_state_dict(torch.load(F"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/Kernel1_conv_{MODEL_SIZE}c.pt"))
loss_fn = nn.CrossEntropyLoss()

# retrieving and preparing testing data
print("[INFO] Loading datasets")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/validation.pt")
print("[INFO] Preparing data")
test_data = sgm.classified_data_slicer(test_data, output_lenght=MODEL_SIZE)

# testing data
print(f"[INFO] Testing on {BL} cubed data, with {MODEL_SIZE} cubed slices") 
time_start = time.time()
nf.test_sliced_data(test_data, model, device)
time_end = time.time()
print(f"time taken {round((time_end - time_start)/60, 2)} mins \n")

print("[INFO] Done! :D")
