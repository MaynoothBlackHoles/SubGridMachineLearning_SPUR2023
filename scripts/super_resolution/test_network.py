"""
Script for testing trained network on chosen dataset
"""

import torch
from torch import nn
import time

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/")

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing
sys.path.append(parent_dir)

from src import network_function as nf
from src import sr_networks as net
from src import subgridmodel as sgm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.Srcnn().to(device)
model.load_state_dict(torch.load(current_dir + f"/network weights/srcnn.pt"))
loss_fn = nn.MSELoss()

# loading and preparing datasets
print("[INFO] Loading datasets")
test_data = torch.load(current_dir + f"/data/validation.pt")
print("[INFO] Batching Data")
test_data = sgm.batch_classified_data(test_data, batch_size=32)

# running test data throgh newtork and evaluating metrics (PSNR) and loss
print("[INFO] Testing Network")
time_start = time.time()
nf.sr_test_loop(test_data, model, loss_fn, device)
time_end = time.time()
print(f"time taken for test: {round((time_end - time_start)/60, 2)} mins \n")

print("[INFO] Done! :D")
