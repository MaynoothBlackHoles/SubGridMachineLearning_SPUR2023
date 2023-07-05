import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import time

import subgridmodel as sgm

class LeNet(torch.nn.Module):
	def __init__(self, numChannels, classes):
		super(LeNet, self).__init__()

		# network structure
		self.stack = nn.Sequential(
			nn.Conv3d(in_channels=numChannels, out_channels=10, kernel_size=2),
			nn.ReLU(),	
			nn.MaxPool3d(kernel_size=2, stride=2),	

			nn.Flatten(),
			nn.Linear(in_features=80, out_features=100),
			nn.ReLU(),

			nn.Linear(in_features=100, out_features=classes),
			nn.LogSoftmax(dim=1)
			)
		
	def forward(self, x):
		# running image through network
		return self.stack(x)
	
	"""def __init__(self, numChannels, classes): # numchannels: 1 for grey scale, 3 for rgb #classes: num of unique classes in data set
		super(LeNet, self).__init__() # call the parent constructor

		# network structure
		self.conv = nn.Conv3d(in_channels=numChannels, out_channels=10, kernel_size=2)
		self.relu1 = nn.ReLU()
		self.maxpol1 = nn.MaxPool3d(kernel_size=2, stride=1)
		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(in_features=8, out_features=100)
		self.relu2 = nn.ReLU()
		self.linear2 = nn.Linear(in_features=100, out_features=classes)
		self.sm = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.conv(x)
		x = self.relu1(x)
		x = self.maxpol1(x)

		x = x = x.view(x.size(0), -1)

		x = self.linear1(x)
		x = self.relu2(x)
		x = self.linear2(x)
		x = self.sm(x)"""

# hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 20
EPOCHS = 5
BOX_LENGTH = 6
DATA_SIZE = 200

print("[INFO] Generating Datasets")
train_data = sgm.gen_batched_classified_data(size=DATA_SIZE, box_lenght=BOX_LENGTH, batch_size=BATCH_SIZE)
test_data = sgm.gen_batched_classified_data(size=DATA_SIZE, box_lenght=BOX_LENGTH, batch_size=BATCH_SIZE)
#validation_data = sgm.gen_classified_data(size=200, box_length=length)
#test_data = sgm.gen_classified_data(size=200, box_length=length)

"""print("[INFO] Loading datasets")
import shelve
dataset = shelve.open('batched_data.db')
train_data = dataset['training']
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet(numChannels=5, classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

def train_loop(dataset):

	for batch, (x, y) in enumerate(dataset):
		(x, y) = (x.to(device), y.to(device))
		
		prediction = model(x)
		loss = loss_fn(prediction, y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss = loss.item()
			print(f"loss: {loss:>7f}")
		
def test_loop(dataset):
    model.eval()

    size = len(dataset)	* BATCH_SIZE
    test_loss = 0
    correct = 0

    with torch.no_grad():
	
        for x, y in dataset:
            pred = model(x)
	    
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss =  (test_loss) / (len(dataset) / BATCH_SIZE)
    correct = correct / (len(dataset) * BATCH_SIZE) # size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for i in range(EPOCHS):
    print(f"[INFO] Epoch {1 + i}")
    train_loop(train_data)
    test_loop(test_data)
