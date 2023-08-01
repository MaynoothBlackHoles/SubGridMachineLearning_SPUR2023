"""
Functions for network training and testing
"""

import torch


def train_loop(dataset, model, loss_fn, device, optimizer, correct_list=[], loss_list=[]):
    """
    Trains network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: cpu or gpu
    opitimiser: optimising function
    correct_list: empty list in which accuracy is stored per epoch
    loss_list: empty list in which loss is stored per epoch
    """
    total_loss = 0
    total_correct = 0
        
    batches = (len(dataset))
    batch_size = len(dataset[0][1])
    size = batches * batch_size
    
    for batch, (x, y) in enumerate(dataset):
		
        if (batch + 1) % 2 == 0:
            percentage = round(100 * ((batch + 1)/batches), 1)
            print(f"\rTraining epoch {percentage}% done", end="")

        (x, y) = (x.to(device), y.to(device))

        prediction = model(x)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        total_loss += loss
        total_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = total_loss / batches
    correct = total_correct / size
    
    print("")
    print(f"Train Error: \n Accuracy: {(correct):.2f}, Avg loss: {avg_loss:.3f} \n")
    correct_list.append(correct)
    loss_list.append(avg_loss)
	
    
    
def test_loop(dataset, model, loss_fn, device, correct_list=[], loss_list=[]):
    """
    Test network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: "cpu" or "cuda", ie cpu or gpu
    correct_list: empty list in which accuracy is stored per epoch
    loss_list: empty list in which loss is stored per epoch
    """
    model.eval()

    total_loss = 0
    total_correct = 0
		
    num_batches = len(dataset)
    batch_size = len(dataset[0][1])
    size = num_batches * batch_size

    with torch.no_grad():
        for x, y in dataset:
            (x, y) = (x.to(device), y.to(device))
            prediction = model(x)
	    
            total_loss += loss_fn(prediction, y).item()
            total_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
	    	
    avg_loss =  total_loss / num_batches
    correct = total_correct / size

    print(f"Test Error: \n Accuracy: {(correct):.2f}, Avg loss: {avg_loss:.3f} \n")
    correct_list.append(correct)
    loss_list.append(avg_loss)
    
    
    
def test_sliced_data(dataset, model, device):
    """
    Tests model on given sliced classified (batched size = 1) dataset

     Variables
    dataset: classified sliced dataset (batch size = 1)
    model: network architecture
    device: "cpu" or "cuda", ie cpu or gpu
    """

    model.eval()
    total_correct = 0

    with torch.no_grad():
        for X, y in dataset: # X is the list of slices, y is the classification of the sliced tensor
            y = y.to(device)

            # loop through slices and if one is star forming then break
            star_forming_slices = 0
            for i, tensor in enumerate(X):
                prediction = model(tensor)
                if prediction.argmax(1) == 1:
                    star_forming_slices += 1
                    break
                 	    
            # check if X is star forming
            if star_forming_slices == int(y):
                total_correct += 1
	    	
    size = len(dataset)
    correct = total_correct / size

    print(f"Test Error: \n Accuracy: {(correct):.2f} \n")



def eval_PSNR(x, y):
    """
    Function to evaluate the PSNR of two tensors; x, y
    """
    MSE = torch.mean(torch.square(x - y))
    MAX_I = torch.max(x)
    PSNR = 20*torch.log10(torch.tensor(MAX_I)) - 10*torch.log10(MSE)
    return float(PSNR)



def test_PSNR(dataset):
    """
    Function to test the PSNR of a dataset which has pairs of tensors one of which has been transformed

    return: average PSNR of dataset
    """

    total_PSNR = 0
        
    batches = (len(dataset))
    
    for batch, (x, y) in enumerate(dataset):
        percentage = round(100 * ((batch + 1)/batches), 1)
        print(f"Training epoch {percentage}% done", end="\r")

        total_PSNR += eval_PSNR(x, y)

    avg_PSNR = total_PSNR / batches
    return avg_PSNR



def sr_train_loop(dataset, model, loss_fn, device, optimizer, PSNR_list=[], loss_list=[]):
    """
    Trains a super resolution network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: cpu or gpu
    opitimiser: optimising function
    PSNR_list: empty list in which average PSNR is stored per epoch
    loss_list: empty list in which average loss is stored per epoch
    """

    total_loss = 0
    total_PSNR = 0
        
    batches = (len(dataset))
    
    for batch, (x, y) in enumerate(dataset):
		
        percentage = round(100 * ((batch + 1)/batches), 1)
        print(f"Training: {percentage}%", end="\r")

        (x, y) = (x.to(device), y.to(device))

        prediction = model(x)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        total_loss += loss
        total_PSNR += eval_PSNR(prediction, y)

    avg_PSNR = total_PSNR / batches
    avg_loss = total_loss / batches

    print(f"Train Error: \n Average PSNR: {(avg_PSNR):.3f}, Avg loss: {avg_loss:.5f} \n")
    loss_list.append(avg_loss)
    PSNR_list.append(avg_PSNR)



def sr_test_loop(dataset, model, loss_fn, device, PSNR_list=[], loss_list=[]):
    """
    Test a super resolution network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: cpu or gpu
    opitimiser: optimising function
    PSNR_list: empty list in which average PSNR is stored per epoch
    loss_list: empty list in which average loss is stored per epoch
    """

    model.eval()

    total_loss = 0
    total_PSNR = 0
        
    batches = (len(dataset))
    
    for batch, (x, y) in enumerate(dataset):
		
        percentage = round(100 * ((batch + 1)/batches), 1)
        print(f"Testing: {percentage}%", end="\r")

        (x, y) = (x.to(device), y.to(device))

        prediction = model(x)
        loss = loss_fn(prediction, y)

        loss = loss.item()
        total_loss += loss
        total_PSNR += eval_PSNR(prediction, y)

    avg_PSNR = total_PSNR / batches
    avg_loss = total_loss / batches

    print(f"Test Error: \n Average PSNR: {(avg_PSNR):.3f}, Avg loss: {avg_loss:.5f} \n")
    loss_list.append(avg_loss)
    PSNR_list.append(avg_PSNR)



def residual_MSELoss(x, convoluted_x, y):
    return torch.mean(torch.square((y - x) - convoluted_x))



def vdsr_train_loop(dataset, model, loss_fn, device, optimizer, PSNR_list=[], loss_list=[]):
    """
    Trains a super resolution network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: cpu or gpu
    opitimiser: optimising function
    PSNR_list: empty list in which average PSNR is stored per epoch
    loss_list: empty list in which average loss is stored per epoch
    """

    total_loss = 0
    total_PSNR = 0
        
    batches = (len(dataset))
    
    for batch, (x, y) in enumerate(dataset):
		
        percentage = round(100 * ((batch + 1)/batches), 1)
        print(f"Training: {percentage}%", end="\r")

        (x, y) = (x.to(device), y.to(device))

        prediction = model(x)
        loss = loss_fn(x, prediction - x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_PSNR += eval_PSNR(prediction, y)

    avg_PSNR = total_PSNR / batches
    avg_loss = total_loss / batches

    print(f"Train Error: \n Average PSNR: {(avg_PSNR):.3f}, Avg loss: {avg_loss:.5f} \n")
    loss_list.append(avg_loss)
    PSNR_list.append(avg_PSNR)



def vdsr_test_loop(dataset, model, loss_fn, device, PSNR_list=[], loss_list=[]):
    """
    Test a super resolution network by running through given dataset

     Variables
    dataset: batched classified dataset  
    model: network architecture
    loss_fn: loss function
    device: cpu or gpu
    opitimiser: optimising function
    PSNR_list: empty list in which average PSNR is stored per epoch
    loss_list: empty list in which average loss is stored per epoch
    """

    model.eval()

    total_loss = 0
    total_PSNR = 0
        
    batches = (len(dataset))
    
    for batch, (x, y) in enumerate(dataset):
		
        percentage = round(100 * ((batch + 1)/batches), 1)
        print(f"Testing: {percentage}%", end="\r")

        (x, y) = (x.to(device), y.to(device))

        prediction = model(x)
        loss = loss_fn(x, prediction - x, y)

        total_loss += float(loss)
        total_PSNR += eval_PSNR(prediction, y)

    avg_PSNR = total_PSNR / batches
    avg_loss = total_loss / batches

    print(f"Test Error: \n Average PSNR: {(avg_PSNR):.3f}, Avg loss: {avg_loss:.5f} \n")
    loss_list.append(avg_loss)
    PSNR_list.append(avg_PSNR)