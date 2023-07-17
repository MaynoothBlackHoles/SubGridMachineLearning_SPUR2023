import torch

def train_loop(dataset, model, loss_fn, device, optimizer, correct_list, loss_list):
	total_loss = 0
	total_correct = 0
        
	batches = (len(dataset))
	batch_size = len(dataset[0][1])
	size = batches * batch_size
    
	for batch, (x, y) in enumerate(dataset):
		
		if (batch + 1) % 2 == 0:
			percentage = round(100 * ((batch + 1)/batches), 1)
			print(f"Training epoch {percentage}% done", end="\r")

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

	print(f"Train Error: \n Accuracy: {(correct):.2f}, Avg loss: {avg_loss:.3f} \n")
	correct_list.append(correct)
	loss_list.append(avg_loss)
	
def test_loop(dataset, model, loss_fn, device, correct_list, loss_list):
    model.eval()

    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for x, y in dataset:
            (x, y) = (x.to(device), y.to(device))
            prediction = model(x)
	    
            total_loss += loss_fn(prediction, y).item()
            total_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
	    	
		
    num_batches = len(dataset)
    batch_size = len(dataset[0][1])
    avg_loss =  total_loss / num_batches
    correct = total_correct / (num_batches * batch_size) # size

    print(f"Test Error: \n Accuracy: {(correct):.2f}, Avg loss: {avg_loss:.3f} \n")
    correct_list.append(correct)
    loss_list.append(avg_loss)
    
def test_(dataset, model,device):
    model.eval()

    total_correct = 0

    with torch.no_grad():
        for X, y in dataset:
            y = y.to(device)

            star_forming_slices = 0
            for i, tensor in enumerate(X):
                prediction = model(tensor)
                if prediction.argmax(1) == 1:
                    star_forming_slices += 1
                    break
                 	    
            #total_loss += loss_fn(prediction, y).item()
            if star_forming_slices == int(y):
                total_correct += 1
	    	
    size = len(dataset)
    correct = total_correct / size

    print(f"Test Error: \n Accuracy: {(correct):.2f} \n")