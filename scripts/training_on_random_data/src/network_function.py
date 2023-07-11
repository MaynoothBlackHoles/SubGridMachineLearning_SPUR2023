import torch

def train_loop(dataset, model, loss_fn, device, optimizer, correct_list, loss_list):
	total_loss = 0
	total_correct = 0
    
	for (x, y) in dataset:
		(x, y) = (x.to(device), y.to(device))
		
		prediction = model(x)
		loss = loss_fn(prediction, y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		loss = loss.item()
		total_loss += loss
		total_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

	batches = (len(dataset))
	batch_size = len(dataset[0][1])
	avg_loss = total_loss / batches
	correct = total_correct / (batches * batch_size)

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