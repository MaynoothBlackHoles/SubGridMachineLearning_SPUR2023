"""
Script used to train networks with chosen architecture and custom datasets
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import sys

import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src import network_function as nf
from src import sr_networks as net
from src import subgridmodel as sgm

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 32 # dont make too big, fails to allocate memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.Srcnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# loading and batching saved datasets from chosen locations
print("[INFO] Loading datasets")
train_data = torch.load(current_dir +  f"/data/training.pt")
test_data = torch.load(current_dir + f"/data/validation.pt")
print("[INFO] Batching Data")
train_data = sgm.batch_classified_data(train_data, BATCH_SIZE)
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

dictionary = {"train PSNR": [], "train loss": [], "test PSNR": [], "test loss": []}

print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    nf.sr_train_loop(train_data, model, loss_fn, device, optimizer, dictionary["train PSNR"], dictionary["train loss"])
    nf.sr_test_loop(test_data, model, loss_fn, device, dictionary["test PSNR"], dictionary["test loss"])

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

    # plot of training info which updates every epoch
    epochs_list = [i for i in range(1, epoch_num + 1)]

    plt.subplot(121)
    plt.plot(epochs_list, dictionary["train PSNR"], label="train PSNR", color="green")
    plt.plot(epochs_list, dictionary["test PSNR"], label="test PSNR", color="red")
    plt.xlabel("Epoch")
    if epoch_num == 1:
        plt.legend()

    plt.subplot(122)
    plt.plot(epochs_list, dictionary["train loss"], "--", label="train loss", color="darkgreen")
    plt.plot(epochs_list, dictionary["test loss"], "--", label="test loss", color="darkred")
    if epoch_num == 1:
        plt.legend()
    plt.xlabel("Epoch")

    plt.savefig("plot")
    # saving network weights 
    torch.save(model.state_dict(), f"srcnn500.pt")
print("[INFO] Done! :D")

plt.show()