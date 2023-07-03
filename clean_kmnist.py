import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time

class LeNet(torch.nn.Module):
	def __init__(self, numChannels, classes): # numchannels: 1 for grey scale, 3 for rgb #classes: num of unique classes in data set
		super(LeNet, self).__init__() # call the parent constructor

		# network structure
		self.stack = nn.Sequential(
			nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2 ,stride=2),

			nn.Flatten(),
			nn.Linear(in_features=800, out_features=500),
			nn.ReLU(),

			nn.Linear(in_features=500, out_features=classes),
			nn.LogSoftmax(dim=1)
			)
		
	def forward(self, x):
		# running image through network
		return self.stack(x)

# hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 1

tick = 0

TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading the dataset...")
trainData = torchvision.datasets.KMNIST(
	root = "data",
	train = True, 
	download = True,
	transform = torchvision.transforms.ToTensor())
testData = torchvision.datasets.KMNIST(
	root = "data",
	train = False,
	download = True,
	transform = torchvision.transforms.ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples], generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
	model.train()
	
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	
	# loop for training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)
		
		# zero out the gradients, perform the backpropagation step,
		# and update the weight
		opt.zero_grad()
		loss.backward()
		opt.step()
		
		# add the loss to the total training loss so far and
		totalTrainLoss += loss
		# calculate the number of correct predictions
		trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # switch off autograd for evaluation
	with torch.no_grad():
		model.eval()
		
		# loop for validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)
			
			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
		
    # calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)
	
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)
	
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))
	
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	
	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)
		
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
		tick += 1
		if tick == 1:
			print(preds)
		
# generate a classification report
from sklearn.metrics import classification_report
print(classification_report(testData.targets.cpu().numpy(), np.array(preds), target_names = testData.classes))

"""from sklearn.metrics import precision_score
print("------------------------------")
newlist = testData.targets.tolist()
print(precision_score(newlist, preds))"""

plt.hist()


# plot the training loss and accuracy
#plt.figure()
"""plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")
plt.show()"""

# serialize the model to disk
torch.save(model, "KMNIST_model.pt")