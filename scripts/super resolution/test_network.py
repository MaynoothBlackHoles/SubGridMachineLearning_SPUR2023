"""

"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import network_function as nf
from src import sr_networks as net
from src import subgridmodel as sgm

BATCH_SIZE = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.Srcnn().to(device)
model.load_state_dict(torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/super resolution/data/srcnn.pt"))
loss_fn = nn.MSELoss()

print("[INFO] Loading datasets")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/super resolution/data/validation.pt")
print("[INFO] Batching Data")
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

dictionary = {"test PSNR": [], "test loss": []}

print("[INFO] Testing Network")
time_start = time.time()
nf.sr_test_loop(test_data, model, loss_fn, device, dictionary["test PSNR"], dictionary["test loss"])
time_end = time.time()
print(f"time taken for test: {round((time_end - time_start)/60, 2)} mins \n")


# saving network weights 
print("[INFO] Done! :D")
