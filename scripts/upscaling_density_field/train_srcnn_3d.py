#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:35:38 2023

@author: john
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from srcnn_3d import SRCNN3, SRDataset3



#%%
current_directory = os.getcwd()
data_path = current_directory + '/../../data/tensors128.npz'

tensors = np.load(data_path)
Xs = [np.log10(X) for X in tensors.values()]

# TODO: expand dataset by factor of 24 via rotations


#%%
print("[INFO] Creating the dataset...")
train_dataset = SRDataset3(Xs, crop=0)


#%%
print("[INFO] Creating the data loaders...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


#%%
model = SRCNN3()


#%%
loss_train_history = []
loss_test_history = []


#%%

# Define loss function and optimizer
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)

learning_rate = 0.0002
layer_specific_lr = [
    {'params': model.layer1.parameters(), 'lr': learning_rate},  
    {'params': model.layer2.parameters(), 'lr': learning_rate},
    {'params': model.layer3.parameters(), 'lr': learning_rate * 0.1},
]

# Create an instance of the SGD optimizer with different learning rates
optimizer = optim.Adam(layer_specific_lr, lr=learning_rate)

#%%
best_loss = 0.020839475854127494

#%%

# Training loop
print("[INFO] Training...")
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()              # Zero the gradients
        outputs = model(inputs)            # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights

        running_loss += loss.item()
        
    # Calculate average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    loss_train_history.append(avg_loss)
    
    print('epoch =', epoch, 
          'train loss =', avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        data = {
            'model_state_dict': model.state_dict(),
            'loss_train_history': loss_train_history,
            'loss_test_history': loss_test_history
        }
        torch.save(data, 'srcnn_model3d_best.pth')
        


#%%
plt.plot(loss_train_history, label='train')
plt.axhline(train_dataset.avg_loss, color='black', label='Tricubic')
# plt.ylim(0.001, 0.01)
plt.legend()
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png', dpi=300)

#%%

#%%
data = {
    'model_state_dict': model.state_dict(),
    'loss_train_history': loss_train_history,
    'loss_test_history': loss_test_history
}
torch.save(data, 'srcnn_model3d_after.pth')

torch.save(optimizer.state_dict(), 'optimizer_after.pth')

#%%
data = torch.load('srcnn_model3d_after.pth')
model.load_state_dict(data['model_state_dict'])
loss_train_history = data['loss_train_history']
loss_test_history  = data['loss_test_history']

optimizer.load_state_dict(torch.load('optimizer_after.pth'))


#%%
train_dataset.compute_bicubic_metrics()

#%%
optimizer.param_groups[0]['lr'] = 0.00005