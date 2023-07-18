"""
Script for loading a trained network and checking its accuracy and loss on chosed dataset
"""

import torch
from torch import nn
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm
from src import network_function as nf
from src import networks as net

BL = 4
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Kernel1_conv(box_length=BL).to(device)
model.load_state_dict(torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/Kernel1_conv_{BL}c.pt"))
loss_fn = nn.CrossEntropyLoss()

print("[INFO] Loading datasets")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/validation.pt")
print("[INFO] Batching Data")
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

dictionary = {"test accuracy": [], "test loss": []}
 
print("[INFO] Testing Network")
nf.test_loop(test_data, model, loss_fn, device, dictionary["test accuracy"], dictionary["test loss"])

print("[INFO] Done! :D")


