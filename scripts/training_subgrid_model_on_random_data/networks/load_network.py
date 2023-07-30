import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir + "/../../..")

import torch
import matplotlib.pyplot as plt

from torch import nn
from subgrid_physics_modelling import data_utils
from subgrid_physics_modelling import networks as nt
from subgrid_physics_modelling import network_training_utils as ntu

# path to directory containing training data
DATA_DIR = current_dir + "/../../../data/random_data/"

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 20
BATCH_SIZE = 64
BL = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nt.Kernel1_conv(box_length=BL).to(device)
model.load_state_dict(torch.load(DATA_DIR + "Kernel1_conv_4c.pt"))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("[INFO] Loading datasets")
train_data = torch.load(DATA_DIR + f"fast_{BL}_cubed/training.pt")
test_data  = torch.load(DATA_DIR + f"fast_{BL}_cubed/validation.pt")

print("[INFO] Batching Data")
train_data = data_utils.batch_classified_data(train_data, BATCH_SIZE)
test_data  = data_utils.batch_classified_data(test_data,  BATCH_SIZE)


stats = {"train accuracy" : [],
         "train loss"     : [],
         "test accuracy"  : [],
         "test loss"      : []}
 
print("[INFO] Training Network")
for i in range(EPOCHS):
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    
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

print("[INFO] Done! :D")
torch.save(model.state_dict(), DATA_DIR + "fcmodel.pt")

epochs = [i for i in range(1, EPOCHS + 1)]
plt.plot(epochs, stats["train accuracy"], label="train accuracy")
plt.plot(epochs, stats["train loss"],     label="train loss",   linestyle="--")
plt.plot(epochs, stats["test accuracy"],  label="test accuracy")
plt.plot(epochs, stats["test loss"],      label="test loss",    linestyle="--")
plt.legend()
plt.savefig("plot")
plt.show()

