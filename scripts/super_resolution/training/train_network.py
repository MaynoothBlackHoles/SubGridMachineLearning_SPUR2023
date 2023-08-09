"""
Script used to train networks with chosen architecture and custom datasets
"""

import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn as nn

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution"

from subgrid_physics_modelling import network_training_utils as ntu
from subgrid_physics_modelling import super_resolution_networks as net
from subgrid_physics_modelling import synthetic_data_generation as sdg

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS        = 40
BATCH_SIZE    = 32

# dataset features
SCALE_FACTOR     = 8
IMAGE_SLICE_SIZE = 32
BIG_TENSORS      = 125

# looking for gpu, if not we use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.CNN_3D(depth        = 3,
                   channels     = 1,
                   mid_channels = 16,
                   kernel_front = 9
                   ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# establishing dataset
print("[INFO] Loading datasets")
dataset = torch.load(DATA_DIR +  f"/datasets/dataset_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
print("[INFO] Batching Data")
dataset["training"] = sdg.batch_classified_data(dataset["training"], BATCH_SIZE, pytorch_hijinks=True)
dataset["validation"] = sdg.batch_classified_data(dataset["validation"], BATCH_SIZE, pytorch_hijinks=True)

# stats to store values
stats = {"train metric": [], "train loss": [], "test metric": [], "test loss": []}
metric_name = "logMSE"

print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    if (i+1) % 10 == 0:
        LEARNING_RATE = LEARNING_RATE * 1e-1
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    # training, testing and evaluating chosen metric and loss
    ntu.sr_train_loop(dataset["training"], model, loss_fn, device, optimizer, stats["train metric"], stats["train loss"], metric_func=ntu.eval_logMSE)
    ntu.sr_test_loop(dataset["validation"], model, loss_fn, device, stats["test metric"], stats["test loss"], metric_func=ntu.eval_logMSE)

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

    # plot of training info which updates every epoch
    epochs_list = [i for i in range(1, epoch_num + 1)]
    ones_list = np.ones(i + 1)

    plt.clf()

    plt.subplot(211)
    plt.plot(epochs_list, stats[f"train metric"], label="train", color="green")
    plt.plot(epochs_list, stats["test metric"], label="test", color="red")
    plt.ylabel(metric_name)
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs_list, stats["train loss"], "--", label="train", color="darkgreen")
    plt.plot(epochs_list, stats["test loss"], "--", label="test", color="darkred")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.savefig(DATA_DIR + f"/plots/plot_cnn_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}")

    # saving model network weights each epoch
    torch.save(model.state_dict(), DATA_DIR + f"/network_weights/rcnn3d_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
    
print("[INFO] Done! :D")
plt.show()