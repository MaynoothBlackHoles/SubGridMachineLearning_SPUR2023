#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:37:05 2023

@author: john
"""

import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir + "/../../..")

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from subgrid_physics_modelling import data_utils
from subgrid_physics_modelling import networks as net
from subgrid_physics_modelling import network_training_utils as ntu
from subgrid_physics_modelling import synthetic_data_generation as sdg

import torch.optim.lr_scheduler as lr_scheduler

# hyperparameters
LEARNING_RATE = 5e-3
EPOCHS        = 100
BATCH_SIZE    = 128
BL            = 6

DATA_DIR = current_dir + "/../../../data/sim_data/"
inputs_filename = DATA_DIR + "my_tensors.npz"
labels_filename = DATA_DIR + "my_labels.npz"


#%% This cell loads and preps simulation data saved in the given files to test
# our model on.
Xs = np.load(inputs_filename)
ls = np.load(labels_filename)

def prep_tensor(X):
    """
    Prepares the given tensor for our pytorch model. The dims of X are permuted
    such that dim 0 conatins the different channels and the others are spatial
    indices.
    
    Also, the channels containing velocity fields are removed and the other
    channels are converted to units that are easier for the model to interpret.
    """
    X = torch.tensor(X).permute(3, 0, 1, 2)
    X = X[[0, 1, 5], :, :, :]
    X[0, :, :, :] /= 1.6e-28
    X[1, :, :, :] /= 1e-6
    return X

test_data = [
              [prep_tensor(Xs[k]) for k in Xs.keys()],
              [torch.tensor(ls[k]) for k in Xs.keys()]
            ]
test_data = data_utils.batch_classified_data(test_data, 1)


#%% This cell generates random data to train our model on.
train_data = sdg.generate_random_compact_object_data(7000, BL)
train_data = data_utils.batch_classified_data(train_data, BATCH_SIZE)


#%% This cell creates our pytorch model.
model   = net.CompactObjectConv()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

stats = {"train accuracy" : [],
          "train loss"    : [],
          "test accuracy" : [],
          "test loss"     : []
          }


#%% This cell initialises the loss function and optimizer used for training.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr = LEARNING_RATE, 
                             weight_decay = 0.000001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=21, gamma=0.5)


#%% This cell computes the initial training stats on the traiing and test data
# before any training has occured.
ntu.test_loop(train_data,
              model,
              loss_fn,
              device,
              stats["train accuracy"],
              stats["train loss"])

ntu.test_loop(test_data,
              model,
              loss_fn,
              device,
              stats["test accuracy"],
              stats["test loss"])


#%% This cell trains the model.

epoch = 0
for i in range(EPOCHS):
    epoch += 1
    print("Training step: ", epoch)
    ntu.train_loop(train_data,
                   model,
                   loss_fn,
                   device,
                   optimizer,
                   stats["train accuracy"],
                   stats["train loss"])
    scheduler.step(epoch)
    
    ntu.test_loop(test_data,
                  model,
                  loss_fn,
                  device,
                  stats["test accuracy"],
                  stats["test loss"])
    
    # Plot training stats
    plt.figure(figsize=(12,6))
    
    plt.subplot(1, 2, 1)
    plt.plot(stats["train accuracy"], label="train accuracy", linewidth=4)
    plt.plot(stats["test accuracy"],  label="test accuracy",  linewidth=4)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Correct classifications over total")
    
    plt.subplot(1, 2, 2)
    plt.plot(stats["train loss"], label="train loss", linewidth=4)
    plt.plot(stats["test loss"],  label="test loss",  linewidth=4)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Loss")
    
    plt.savefig("plot.png", dpi=300)
    plt.close()

    # saving network weights 
    torch.save(model.state_dict(), DATA_DIR + f"BH_subgrid_model_ran_{BL}c.pt")
    
print("[INFO] Done! :D")
plt.show()