import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data")
from src import subgridmodel as sgm
from src import network_function as nf
from src import networks as net

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 20
BATCH_SIZE = 64
BL = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Kernel1_conv_big(box_length=BL).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("[INFO] Loading datasets")
train_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/torch data/fast {BL} cubed/training.pt")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/torch data/fast {BL} cubed/validation.pt")
print("[INFO] Batching Data")
train_data = sgm.batch_classified_data(train_data, BATCH_SIZE)
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

dictionary = {"train accuracy": [], "train loss": [], "test accuracy": [], "test loss": []}
 
print("[INFO] Training Network")
for i in range(EPOCHS):
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    time_start = time.time()

    nf.train_loop(train_data, model, loss_fn, device, optimizer, dictionary["train accuracy"], dictionary["train loss"])
    nf.test_loop(test_data, model, loss_fn, device, dictionary["test accuracy"], dictionary["test loss"])

    time_end = time.time()
    print(f"time taken for epoch {round((time_end - time_start)/60, 2)} mins \n")

epochs_list = [i for i in range(1, EPOCHS + 1)]
plt.plot(epochs_list, dictionary["train accuracy"], label="train accuracy", color="green")
plt.plot(epochs_list, dictionary["train loss"], "--", label="train loss", color="darkgreen")
plt.plot(epochs_list, dictionary["test accuracy"], label="test accuracy", color="red")
plt.plot(epochs_list, dictionary["test loss"], "--", label="test loss", color="darkred")
plt.legend()
plt.xlabel("Epoch")
plt.savefig("plot")

print("[INFO] Done! :D")

plt.show()
#torch.save(model, "C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/trained_networks/Kernel1_conv.pt")
