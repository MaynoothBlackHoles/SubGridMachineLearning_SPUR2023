#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:38:25 2023

@author: john
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from EDRS3D import EDSR
from EDSR3D_dataset import EDSR3DDataset
from torch.utils.data import DataLoader



#%%
model = EDSR()
# data = torch.load('srcnn_model3d.pth')
# model.load_state_dict(data['model_state_dict'])
# loss_train_history = data['loss_train_history']
# loss_test_history  = data['loss_test_history']


#%%
current_directory = os.getcwd()
data_path = current_directory + '/../../data/tensors128.npz'

tensors = np.load(data_path)
Xs = [np.log10(X) for X in tensors.values()]

# TODO: expand dataset by factor of 24 via rotations


#%%
print("[INFO] Creating the dataset...")
train_dataset = EDSR3DDataset(Xs)


#%%
X_orig = train_dataset.regions[-2]


X_down = train_dataset.rescale([X_orig], 1/4)[0]
X_bi = train_dataset.rescale([X_down], 4)[0]

#%%
X_sr = model(X_down)[0, :, :, :, :]

#%%
criterion = torch.nn.L1Loss()
loss_bi = criterion(X_bi, X_orig) 
loss_sr = criterion(X_sr, X_orig) 


#%%

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(np.log10(np.sum(10**X_orig.numpy()[0, :, :, :], axis=2)))
ax1.set_title('Original')
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(np.log10(np.sum(10**X_down.numpy()[0, :, :, :], axis=2)))
ax2.set_title('down scaled')
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(np.log10(np.sum(10**X_bi.numpy()[0, :, :, :], axis=2)))
ax3.set_title(f'Tricubic {loss_bi:.6f}')
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(np.log10(np.sum(10**X_sr.detach().numpy()[0, :, :, :], axis=2)))
ax4.set_title(f'Network Model {loss_sr:.6f}')
ax4.set_xticks([])
ax4.set_yticks([])

plt.subplots_adjust(wspace = 0.0, hspace=0.1)

plt.savefig('test1.png', dpi=300)