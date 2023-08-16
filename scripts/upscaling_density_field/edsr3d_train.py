#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:31:40 2023

@author: john
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from EDRS3D import EDSR
from EDSR3D_dataset import EDSR3DDataset, expand_dataset, DensityLoss
from torch.utils.data import DataLoader


#%%
current_directory = os.getcwd()
data_path = current_directory + '/../../data/tensors128.npz'

tensors = np.load(data_path)
Xs = [np.log10(X) for X in tensors.values()]
# Xs = expand_dataset(Xs)


#%%
print("[INFO] Creating the dataset...")
train_dataset = EDSR3DDataset(Xs)


#%%
print("[INFO] Creating the data loaders...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


#%%
model = EDSR()


#%%
loss_train_history = []
loss_test_history = []


#%%

# Define loss function and optimizer
criterion = DensityLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
#%%
learning_rate = 0.0001

# Create an instance of the SGD optimizer with different learning rates
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
best_loss = 10

#%%

# Training loop
print("[INFO] Training...")
num_epochs = 10
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
        torch.save(data, 'edsr_model3d_best.pth')
        


#%%
plt.plot(loss_train_history, label='train')
plt.axhline(train_dataset.avg_dl, color='black', label='Tricubic')
# plt.ylim(0.001, 0.01)
plt.legend()
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png', dpi=300)


#%%
data = {
    'model_state_dict': model.state_dict(),
    'loss_train_history': loss_train_history,
    'loss_test_history': loss_test_history
}
torch.save(data, 'edsr_model3d_after.pth')

torch.save(optimizer.state_dict(), 'optimizer_after.pth')


#%%
data = torch.load('edsr_model3d_after.pth')
model.load_state_dict(data['model_state_dict'])
loss_train_history = data['loss_train_history']
loss_test_history  = data['loss_test_history']

optimizer.load_state_dict(torch.load('optimizer_after.pth'))


#%%
train_dataset.compute_bicubic_metrics()


#%%
optimizer.param_groups[0]['lr'] = 0.00005