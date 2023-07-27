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
current_dir = current_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/")
sys.path.append(parent_dir)

from src import network_function as nf
from src import sr_networks as net
from src import subgridmodel as sgm

# hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 256

# dataset features
IMAGE_SLICE_SIZE = 33
SCALE_FACTOR = 2

# looking for gpu, if not we use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using {device}")

# loadng network architecture, choosing optimiser and loss function
model = net.Srcnn().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# establishing dataset
print("[INFO] Loading datasets")
dataset = torch.load(current_dir +  f"/data/dataset_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
print("[INFO] Batching Data")
dataset["training"] = sgm.batch_classified_data(dataset["training"], BATCH_SIZE)
dataset["validation"] = sgm.batch_classified_data(dataset["validation"], BATCH_SIZE)

# dictionary to store values
dictionary = {"train PSNR": [], "train loss": [], "test PSNR": [], "test loss": []}

print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    # training, testing and evaluating chosen metric (PSNR) and loss
    nf.sr_train_loop(dataset["training"], model, loss_fn, device, optimizer, dictionary["train PSNR"], dictionary["train loss"])
    nf.sr_test_loop(dataset["validation"], model, loss_fn, device, dictionary["test PSNR"], dictionary["test loss"])

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

    # plot of training info which updates every epoch
    epochs_list = [i for i in range(1, epoch_num + 1)]
    ones_list = np.ones(i + 1)

    plt.clf()
    plt.subplot(211)
    plt.plot(epochs_list, dictionary["train PSNR"], label="train", color="green")
    plt.plot(epochs_list, dictionary["test PSNR"], label="test", color="red")
    plt.plot(epochs_list, ones_list * 29.3, label="bicubic", color="blue")
    plt.ylabel("PSNR")
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs_list, dictionary["train loss"], "--", label="train", color="darkgreen")
    plt.plot(epochs_list, dictionary["test loss"], "--", label="test", color="darkred")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("plot_srcnn")

    # saving model network weights each epoch
    torch.save(model.state_dict(), f"srcnn_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
    
print("[INFO] Done! :D")
plt.show()