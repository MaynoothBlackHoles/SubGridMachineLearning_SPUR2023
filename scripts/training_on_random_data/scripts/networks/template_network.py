import torch
from torch import nn
import matplotlib.pyplot as plt
import shelve

import sys
sys.path.append("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data")
from src import subgridmodel as sgm
from src import network_function as nf

class myNet(torch.nn.Module):
	def __init__(self, numChannels, classes):
		super(myNet, self).__init__()

		# network structure
		self.stack = nn.Sequential(
			#nn.Conv3d(in_channels=numChannels, out_channels=5, kernel_size=3, stride=1),
			#nn.ReLU(),	
			#nn.MaxPool3d(kernel_size=2, stride=2),	

			#nn.Conv3d(in_channels=5, out_channels=5, kernel_size=2, stride=1),
			#nn.ReLU(),	
			#nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Flatten(),
			nn.Linear(in_features=4**3*5, out_features=1000),
			nn.ReLU(),

			nn.Linear(in_features=1000, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		# running image through network
		return self.stack(x)

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 3
BATCH_SIZE = 64

print("[INFO] Loading datasets")
dataset = shelve.open('C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast 4 cubed/fast_4cubed.db')
train_data = dataset['training']
test_data = dataset['validation']
print("[INFO] Batching Data")
train_data = sgm.batch_classified_data(train_data, BATCH_SIZE)
test_data = sgm.batch_classified_data(test_data, BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = myNet(numChannels=5, classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

dictionary = {"train accuracy": [], "train loss": [], "test accuracy": [], "test loss": []}
 
print("[INFO] Training Network")
for i in range(EPOCHS):
    print(f"[INFO] Epoch {i + 1} ---------------------------")
    nf.train_loop(train_data, model, loss_fn, device, optimizer, dictionary["train accuracy"], dictionary["train loss"])
    nf.test_loop(test_data, model, loss_fn, device, dictionary["test accuracy"], dictionary["test loss"])

print("[INFO] Done! :D")
#torch.save(model, "model.pt")

epochs_list = [i for i in range(1, EPOCHS + 1)]
plt.plot(epochs_list, dictionary["train accuracy"], label="train accuracy", color="green")
plt.plot(epochs_list, dictionary["train loss"], "--", label="train loss", color="darkgreen")
plt.plot(epochs_list, dictionary["test accuracy"], label="test accuracy", color="red")
plt.plot(epochs_list, dictionary["test loss"], "--", label="test loss", color="darkred")
plt.legend()
plt.xlabel("Epoch")
#plt.savefig("plot")
plt.show()

