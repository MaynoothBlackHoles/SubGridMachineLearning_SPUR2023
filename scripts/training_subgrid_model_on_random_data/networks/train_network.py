"""
Script used to train networks with chosen architecture and datasets
"""

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/random_data"

import time
import torch
from torch import nn
import matplotlib.pyplot as plt

from subgrid_physics_modelling import data_utils
from subgrid_physics_modelling import networks as net
from subgrid_physics_modelling import network_training_utils as ntu

# path to directory containing training data
DATA_DIR = top_dir + "/data/random_data/"

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS        = 100
BATCH_SIZE    = 64
BL            = 8

# loadng network architecture, choosing optimiser and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.Kernel1_conv(box_length=BL).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# loading and batching saved datasets from chosen locations
print("[INFO] Loading datasets")
train_data = torch.load(DATA_DIR + f"/datasets/training_{BL}.pt")
test_data  = torch.load(DATA_DIR + f"/datasets/validation_{BL}.pt")

print("[INFO] Batching Data")
train_data = data_utils.batch_classified_data(train_data, BATCH_SIZE)
test_data  = data_utils.batch_classified_data(test_data,  BATCH_SIZE)

stats = {"train accuracy" : [], 
         "train loss"     : [], 
         "test accuracy"  : [], 
         "test loss"      : []}

# parameters used for plotting the above stats.
COLORS = ["green", "darkgreen", "red", "darkred"]
STYLES = ["-", "--", "-", "--"]

print("[INFO] Training Network")
for epoch in range(1, EPOCHS+1):
    print(f"[INFO] Epoch {epoch} ---------------------------")
    time_start = time.time()

    # training and evaluating network accuracy and loss
    ntu.train_loop(train_data, 
                   model, 
                   loss_fn, 
                   device, 
                   optimizer, 
                   stats["train accuracy"], 
                   stats["train loss"])
    
    ntu.test_loop(test_data, 
                  model, 
                  loss_fn, 
                  device, 
                  stats["test accuracy"], 
                  stats["test loss"])

    time_end = time.time()
    time_taken = round((time_end - time_start)/60, 2)
    print(f"time taken for epoch {time_taken} mins \n")
    
    epochs = [i for i in range(1, epoch + 1)]
    for (name, data), color, style in zip(stats.items(), COLORS, STYLES):
        plt.plot(epochs, data, label=name, color=color, linestyle=style)
    
    if epoch == 1:
        plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(DATA_DIR + f"/plots/Kernel1_conv_{BL}c")

    # saving network weights 
    torch.save(model.state_dict(), DATA_DIR + f"/network_weights/Kernel1_conv_{BL}c.pt")
    
print("[INFO] Done! :D")
plt.show()