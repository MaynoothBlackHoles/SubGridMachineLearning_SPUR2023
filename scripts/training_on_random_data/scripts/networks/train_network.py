import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm
from src import network_function as nf
from src import networks as net

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 64
BL = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Kernel1_conv(box_length=BL).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("[INFO] Loading datasets")
train_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/training.pt")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/validation.pt")
print("[INFO] Batching Data")
train_data = sgm.batch_classified_data(train_data, BATCH_SIZE)
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

dictionary = {"train accuracy": [], "train loss": [], "test accuracy": [], "test loss": []}
 
print("[INFO] Training Network")
epoch_num = 0
for i in range(EPOCHS):
    epoch_num += 1
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    nf.train_loop(train_data, model, loss_fn, device, optimizer, dictionary["train accuracy"], dictionary["train loss"])
    nf.test_loop(test_data, model, loss_fn, device, dictionary["test accuracy"], dictionary["test loss"])

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

    epochs_list = [i for i in range(1, epoch_num + 1)]
    plt.plot(epochs_list, dictionary["train accuracy"], label="train accuracy", color="green")
    plt.plot(epochs_list, dictionary["train loss"], "--", label="train loss", color="darkgreen")
    plt.plot(epochs_list, dictionary["test accuracy"], label="test accuracy", color="red")
    plt.plot(epochs_list, dictionary["test loss"], "--", label="test loss", color="darkred")
    if epoch_num == 1:
        plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("plot")

    torch.save(model.state_dict(), f"Kernel1_conv_{BL}c.pt") #C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/trained_networks/
print("[INFO] Done! :D")

plt.show()
    