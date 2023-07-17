import torch
from torch import nn
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm
from src import network_function as nf
from src import networks as net

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 100
BATCH_SIZE = 64
BL = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Kernel1_conv(box_length=BL).to(device)
model.load_state_dict(torch.load("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/Kernel1_conv_4c.pt"))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("[INFO] Loading datasets")
train_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/{BL} cubed/testing.pt")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/{BL} cubed/validation.pt")
print("[INFO] Batching Data")
train_data = sgm.batch_classified_data(train_data, BATCH_SIZE)
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)


dictionary = {"train accuracy": [], "train loss": [], "test accuracy": [], "test loss": []}
 
print("[INFO] Training Network")
for i in range(EPOCHS):
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    nf.train_loop(train_data, model, loss_fn, device, optimizer, dictionary["train accuracy"], dictionary["train loss"])
    nf.test_loop(test_data, model, loss_fn, device, dictionary["test accuracy"], dictionary["test loss"])

print("[INFO] Done! :D")
torch.save(model.state_dict(), "Kernel1_conv.pt")

epochs_list = [i for i in range(1, EPOCHS + 1)]
plt.plot(epochs_list, dictionary["train accuracy"], label="train accuracy")
plt.plot(epochs_list, dictionary["train loss"], "--", label="train loss")
plt.plot(epochs_list, dictionary["test accuracy"], label="test accuracy")
plt.plot(epochs_list, dictionary["test loss"], "--", label="test loss")
plt.legend()
plt.savefig("plot")
plt.show()