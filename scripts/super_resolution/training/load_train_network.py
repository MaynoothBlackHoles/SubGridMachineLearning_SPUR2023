"""
Script used to train networks with chosen architecture and custom datasets
"""

import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn as nn
import random

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
LEARNING_RATE = 1e-3
EPOCHS        = 40
BATCH_SIZE    = 16

# dataset features
SCALE_FACTOR     = 4
IMAGE_SLICE_SIZE = 32
BIG_TENSORS      = 16

#bicubic_mse = {"sf 4": 1, "sf 8": 32, "sf 16": 22}

# looking for gpu, if not we use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loadng network architecture, choosing optimiser and loss function
model = net.CNN_3D(depth=3, channels=1, mid_channels=16, kernel_front=9)

new_dict = {}
for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    shape = model.state_dict()[param_tensor].size()
    tuple_shape = tuple(shape)
    identity_maps = torch.zeros(*tuple_shape,)
    if len(shape) > 1:
        for i in range(tuple_shape[0]):
            for j in range(tuple_shape[1]): 
                identity_maps[i, j, 1, 1 ,1] = 1/10

    new_dict[param_tensor] = identity_maps

model.load_state_dict(new_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# establishing dataset
print("[INFO] Loading datasets")
dataset = torch.load(DATA_DIR +  f"/datasets/dataset_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
dataset_copy = dataset
print("[INFO] Batching Data")
#dataset["training"] = sdg.batch_classified_data(dataset["training"], BATCH_SIZE, pytorch_hijinks=True)
dataset["validation"] = sdg.batch_classified_data(dataset["validation"], BATCH_SIZE, pytorch_hijinks=True)

# stats to store values
stats = {"train metric": [], "train loss": [], "test metric": [], "test loss": []}
metric_name = "logMSE"

print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    print("Shuffling training data")
    training_data = dataset_copy["training"]
    scaled, origial = training_data[0], training_data[1]

    c = list(zip(scaled, origial))
    random.shuffle(c)
    scaled, origial = zip(*c)

    training = [scaled, origial]

    print("Batching Data")
    training = sdg.batch_classified_data(training, BATCH_SIZE, pytorch_hijinks=True)

    # training, testing and evaluating chosen metric and loss
    ntu.sr_train_loop(training, model, loss_fn, device, optimizer, stats["train metric"], stats["train loss"], metric_func=ntu.eval_logMSE)
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
    #plt.plot(epochs_list, ones_list * bicubic_mse[f"sf {SCALE_FACTOR}"], label="test", color="red")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.savefig(DATA_DIR + f"/plots/plot_idcnn_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}")

    # saving model network weights each epoch
    torch.save(model.state_dict(), DATA_DIR + f"/network_weights/idcnn3d_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
    
print("[INFO] Done! :D")
plt.show()