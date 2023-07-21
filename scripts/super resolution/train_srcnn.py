"""
Script used to train networks with chosen architecture and custom datasets
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import numpy as np

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
BATCH_SIZE = 128

IMAGE_SIZE = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.Srcnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# loading and batching saved datasets from chosen locations
print("[INFO] Loading datasets")
train_data = torch.load(current_dir +  f"/data/training_{IMAGE_SIZE}s.pt")
test_data = torch.load(current_dir + f"/data/validation_{IMAGE_SIZE}s.pt")
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
    ones_list = np.ones(i + 1)

    plt.subplot(121)
    plt.plot(epochs_list, dictionary["train PSNR"], label="train", color="green")
    plt.plot(epochs_list, dictionary["test PSNR"], label="test", color="red")
    plt.plot(epochs_list, ones_list * 81.25, label="bicubic", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    if epoch_num == 1:
        plt.legend()

    plt.subplot(122)
    plt.plot(epochs_list, dictionary["train loss"], "--", label="train", color="lightgreen")
    plt.plot(epochs_list, dictionary["test loss"], "--", label="test", color="lightred")
    if epoch_num == 1:
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("plot")
    torch.save(model.state_dict(), f"srcnn{IMAGE_SIZE}.pt")
    
print("[INFO] Done! :D")

plt.show()