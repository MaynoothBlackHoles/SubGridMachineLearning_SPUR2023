"""
Script used to train networks with chosen architecture and custom datasets
"""

import torch
import matplotlib.pyplot as plt
import time
import numpy as np

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution/datasets"

from subgrid_physics_modelling import network_training_utils as ntu
from subgrid_physics_modelling import super_resolution_networks as net
from subgrid_physics_modelling import synthetic_data_generation as sdg


# hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 20
BATCH_SIZE = 128

SIZE = 100
IMAGE_SLICE_SIZE = 33
SCALE_FACTOR = 2

# looking for gpu, if not we use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# loadng network architecture, choosing optimiser and loss function
model = net.VDSRcnn(depth=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = ntu.residual_MSELoss

# establishing dataset
print("[INFO] Loading datasets")
dataset = torch.load(current_dir +  f"/data/dataset_{SIZE}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
print("[INFO] Batching Data")
dataset["training"] = sdg.batch_classified_data(dataset["training"], BATCH_SIZE)
dataset["validation"] = sdg.batch_classified_data(dataset["validation"], BATCH_SIZE)

# dictionary to store values
dictionary = {"train PSNR": [], "train loss": [], "test PSNR": [], "test loss": []}
sf2_PSNR = []
sf4_PSNR = []
sf8_PSNR = []

print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    # training, testing and evaluating chosen metric (PSNR) and loss
    ntu.vdsr_train_loop(dataset["training"], model, loss_fn, device, optimizer, dictionary["train PSNR"], dictionary["train loss"])
    ntu.vdsr_test_loop(dataset["validation"], model, loss_fn, device, dictionary["test PSNR"], dictionary["test loss"])

    ntu.vdsr_test_loop(dataset["single sf data"][0], model, loss_fn, device, sf2_PSNR)
    ntu.vdsr_test_loop(dataset["single sf data"][1], model, loss_fn, device, sf4_PSNR)
    ntu.vdsr_test_loop(dataset["single sf data"][2], model, loss_fn, device, sf8_PSNR)

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

    # plot of training info which updates every epoch
    epochs_list = [i for i in range(1, epoch_num + 1)]
    ones_list = np.ones(i + 1)

    plt.clf()
    plt.subplot(211)
    plt.plot(epochs_list, dictionary["train PSNR"], label="train", color="green")
    plt.plot(epochs_list, dictionary["test PSNR"], label="test", color="red")
    
    plt.plot(epochs_list, sf2_PSNR, "--", label="sf2")
    plt.plot(epochs_list, sf4_PSNR, "--", label="sf2")
    plt.plot(epochs_list, sf8_PSNR, "--", label="sf2")

    plt.ylabel("PSNR")
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs_list, dictionary["train loss"], "--", label="train", color="darkgreen")
    plt.plot(epochs_list, dictionary["test loss"], "--", label="test", color="darkred")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("plot_vdsrcnn")

    # saving model network weights each epoch
    torch.save(model.state_dict(), f"vdsrcnn_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}.pt")
    
print("[INFO] Done! :D")
plt.show()