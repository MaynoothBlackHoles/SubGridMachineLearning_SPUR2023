#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:58:57 2023

@author: john
"""
import os
import numpy as np
import torchvision

from PIL import Image
from scipy.ndimage import gaussian_filter, zoom



#%%
current_directory = os.getcwd()
data_dir = current_directory + '/../../../data/flowers'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


print("[INFO] loading the dataset...")
trainData = torchvision.datasets.Flowers102(root     = data_dir, 
                                            split    = "train",
                                            download = True)

valData = torchvision.datasets.Flowers102(root     = data_dir,
                                          split    = "val",
                                          download = True)

testData = torchvision.datasets.Flowers102(root     = data_dir, 
                                           split    = "test",
                                           download = True)



#%%
original_image = trainData[22][0]
original_array = np.asarray(original_image)



#%% Apply gaussian filter
sigma = 2

filtered_array = np.zeros_like(original_array)
for channel in range(3):
    original_channel = original_array[:, :, channel]
    filtered_array[:, :, channel] = gaussian_filter(original_channel, sigma)
    
filtered_image = Image.fromarray(filtered_array)



#%% Downscale the filtered image
scaling_factor = 1/4

downscaled_array = []
for channel in range(3):
    downscaled_array.append(zoom(filtered_array[:, :, channel],
                                 (scaling_factor, scaling_factor), 
                                 order=3)
                            )
downscaled_array = np.stack(downscaled_array, axis=2)
downscaled_image = Image.fromarray(downscaled_array)



#%% Downscale the original image for comparison
downscaled_original = []
for channel in range(3):
    downscaled_original.append(zoom(original_array[:, :, channel],
                                    (scaling_factor, scaling_factor), 
                                    order=3)
                               )
downscaled_original = np.stack(downscaled_original, axis=2)
downscaled_original_image = Image.fromarray(downscaled_original)



#%% Upscale the filtered image to the original size.
scaling_factor = 4

upscaled_array = []
for channel in range(3):
    upscaled_array.append(zoom(downscaled_array[:, :, channel],
                               (scaling_factor, scaling_factor), 
                               order=3)
                         )

upscaled_array = np.stack(upscaled_array, axis=2)
upscaled_image = Image.fromarray(upscaled_array)



#%% Upscale the original image to the original size.
scaling_factor = 4

upscaled_original = []
for channel in range(3):
    upscaled_original.append(zoom(downscaled_original[:, :, channel],
                               (scaling_factor, scaling_factor), 
                               order=3)
                         )

upscaled_original = np.stack(upscaled_original, axis=2)
upscaled_original_image = Image.fromarray(upscaled_original)



#%% Compute the mean squared error between upscaled images and original.
x, y, z = np.minimum(upscaled_array.shape, original_array.shape)
mse_filter = np.mean(np.square(original_array[:x, :y, :] - upscaled_array[:x, :y, :]))

x, y, z = np.minimum(upscaled_original.shape, original_array.shape)
mse_original = np.mean(np.square(original_array[:x, :y, :] - upscaled_original[:x, :y, :]))

MAX_I = np.max(original_array)
PSNR_filter = 20 * np.log10(MAX_I) - 10 * np.log10(mse_filter)
PSNR_original = 20 * np.log10(MAX_I) - 10 * np.log10(mse_original)



#%% print results
print(f'The PSNR for the filtered reconstruction is {PSNR_filter} dB')
print(f'The PSNR for the reconstruction with no filtering is {PSNR_original} dB')

# #%%
# original_image.show()
# #%%
# upscaled_image.show()
# #%%
# upscaled_original_image.show()