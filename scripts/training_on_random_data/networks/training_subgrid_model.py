#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:37:05 2023

@author: john
"""

import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)
parent_dir = parent_dir.replace("\\", "/")

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from subgrid_physics_modelling import data_utils
from subgrid_physics_modelling import networks as net
from subgrid_physics_modelling import network_training_utils as ntu
from subgrid_physics_modelling import synthetic_data_generation as sdg

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS        = 100
BATCH_SIZE    = 35
BL            = 4

DATA_DIR = current_dir + "/../../../data/sim_data/"
inputs_filename = DATA_DIR + "my_tensors.npz"
labels_filename = DATA_DIR + "my_labels.npz"


#%%
Xs = np.load(inputs_filename)
ls = np.load(labels_filename)

test_data = [
             [torch.tensor(Xs[k]).permute(3, 0, 1, 2) for k in Xs.keys()],
             [torch.tensor(ls[k]) for k in Xs.keys()]
            ]
test_data = data_utils.batch_classified_data(test_data, 1)


#%%
train_data = sdg.generate_random_compact_object_data(1400, 4)


#%%
# train_data = torch.load(DATA_DIR + f"fast_{BL}_cubed/training.pt")
train_data = data_utils.batch_classified_data(train_data, BATCH_SIZE)


#%%
model   = net.CompactObjectConv(numChannels = 6, classes = 4)
loss_fn = nn.CrossEntropyLoss()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

stats = {"train accuracy" : [],
         "train loss"     : [],
         "test accuracy"  : [],
         "test loss"      : []
         }


#%%

# parameters used for plotting the above stats.
COLORS = ["green", "darkgreen", "red", "darkred"]
STYLES = ["-", "--", "-", "--"]

for epoch in range(1, EPOCHS+1):
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
    
    epochs = [i for i in range(1, epoch + 1)]
    for (name, data), color, style in zip(stats.items(), COLORS, STYLES):
        plt.plot(data, label=name, color=color, linestyle=style)
    
    if epoch == 1:
        plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("plot")

    # saving network weights 
    torch.save(model.state_dict(), DATA_DIR + f"BH_subgrid_model_ran_{BL}c.pt")
    
print("[INFO] Done! :D")
plt.show()