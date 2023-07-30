import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import torch
import time

dropout_p = .25

class myNet(torch.nn.Module):
	def __init__(self, numChannels, classes): # numchannels: 1 for grey scale, 3 for rgb #classes: num of unique classes in data set
		# call the parent constructor
		super(myNet, self).__init__()

		# network structure
		self.stack = nn.Sequential(
			#nn.Dropout2d(dropout_p),
			nn.Conv2d(in_channels=numChannels, out_channels=60, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=3),

			#nn.Dropout2d(dropout_p),
			nn.Conv2d(in_channels=60, out_channels=120, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3 ,stride=3),

			nn.Flatten(),
			#nn.Dropout(dropout_p),
			nn.Linear(in_features=480, out_features=1000),
			nn.ReLU(),

			nn.Linear(in_features=1000, out_features=classes),
			nn.LogSoftmax(dim=1)
			)
		
	def forward(self, x):
		# running image through network
		return self.stack(x)

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define training hyperparameters
INIT_LR = .001
BATCH_SIZE = 64
EPOCHS = 3
IMAGE_SIZE = 100

# initialize our model, optimizer and loss function
print("[INFO] initializing the LeNet model...")
model = myNet(numChannels=3, classes=102).to(device)
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=.01)
lossFn = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] Number of parameters: {total_params}") # 406502

# transfrom functions since the flowers images are all different sizes
data_transform_train = transforms.Compose([
	transforms.CenterCrop(500),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(10),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

data_transform = transforms.Compose([
	transforms.CenterCrop(500),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# load the datasets
print("[INFO] loading the dataset...")
trainData = torchvision.datasets.Flowers102(
	root="data", 
	split="train", 
	download=True,
	transform=data_transform_train)

valData = torchvision.datasets.Flowers102(
	root="data", 
	split="val", 
	download=True,
	transform=data_transform)

testData = torchvision.datasets.Flowers102(
	root="data",
    split="test", 
    download=True,
	transform=data_transform)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

"""print("[INFO] Size of data sets")  Not sure what this is printing out
print(f"training data: {len(trainDataLoader.dataset)}")
print(f"validatoin data: {len(valDataLoader.dataset)}")
print(f"test data: {len(testDataLoader.dataset)}")"""

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE


# dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": []}

print("[INFO] training the network...")
startTime = time.time() 
# loop over epochs
for e in range(0, EPOCHS):
	model.train() # training 
	
	e_startTime = time.time()

	totalTrainLoss = 0
	trainCorrect = 0

	totalValLoss = 0
	valCorrect = 0
	
	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)
		
		opt.zero_grad() # zero out the gradients
		loss.backward() # perform the backpropagation step
		opt.step() # update the weights
		
		totalTrainLoss += loss # add the loss to the total training loss so far and
		trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item() # calculate the number of correct predictions

    # switch off autograd for evaluation
	with torch.no_grad():
		model.eval() # evaluation mode
		
		# loop over the validation set
		for (x, y) in valDataLoader:

			(x, y) = (x.to(device), y.to(device)) # send the input to the device
			
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
	e_endTime = time.time()
	print("Time to run epoch: {:2f}s".format(e_endTime - e_startTime))
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
print("[INFO] checking model with test data...")
with torch.no_grad():
	model.eval()
	preds = [] # list to store predictions
	
	
	for (x, y) in testDataLoader:

			(x, y) = (x.to(device), y.to(device)) # send the input to the device
			
			# make the predictions and calculate the validation loss
			pred = model(x)
			
			# calculate the number of correct predictions
			pred = model(x)
			preds.extend(pred.argmax(axis=1).cpu().numpy())


#print("[INFO] Final test accuracy: {}".format(H["test_acc"][-1]))

# plot the training loss and accuracy
#plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.grid()
plt.savefig("plot")
plt.show()

#torch.save(model, "model")