#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:53:05 2023

@author: john
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init
from scipy.ndimage import zoom
from torch.utils.data import Dataset



class EDSR(torch.nn.Module):
    """
    Note, kernel size should be an odd number.
    """
    
    def __init__(self, num_res_blocks = 2, filters = 16, kernel_size = 3):
        
        super().__init__()
        
        self.layerA = nn.Conv3d(in_channels  = 1, 
                                out_channels = filters, 
                                kernel_size  = kernel_size, 
                                stride       = 1,
                                padding      = (kernel_size-1)//2,
                                dtype=torch.float32)
        
        self.res_blocks = [ResBlock(filters, kernel_size) 
                           for i in range(num_res_blocks)]
        
        self.layerB = nn.Conv3d(in_channels  = filters, 
                                out_channels = filters, 
                                kernel_size  = kernel_size, 
                                stride       = 1,
                                padding      = (kernel_size-1)//2,
                                dtype=torch.float32)
        
        self.upsample_layer1 = Upsampler(filters, kernel_size)
        self.upsample_layer2 = Upsampler(filters, kernel_size)
        
        self.layerC = nn.Conv3d(in_channels  = filters, 
                                out_channels = 1, 
                                kernel_size  = kernel_size, 
                                stride       = 1,
                                padding      = (kernel_size-1)//2,
                                dtype=torch.float32)
        
    def forward(self, x_in):
        
        # Convolutional head
        x = self.layerA(x_in)
        x_skip = x
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Middle convolutional layer
        x = self.layerB(x)
        x = x + x_skip
        
        # Scale up
        x = self.upsample_layer1(x)
        x = self.upsample_layer2(x)
        x = self.layerC(x)
        
        return x



class ResBlock(torch.nn.Module):
    
    def __init__(self, channels, kernel_size):
        super().__init__()
        
        self.layer1 = nn.Conv3d(in_channels  = channels, 
                                out_channels = channels, 
                                kernel_size  = kernel_size, 
                                stride       = 1,
                                padding      = (kernel_size-1)//2,
                                dtype=torch.float32)
        self.relu = nn.ReLU()
        
        self.layer2 = nn.Conv3d(in_channels  = channels, 
                                out_channels = channels, 
                                kernel_size  = kernel_size, 
                                stride       = 1,
                                padding      = (kernel_size-1)//2,
                                dtype=torch.float32)
        
    def forward(self, x_in):
        
        x = self.layer1(x_in)
        x = self.relu(x)
        x = self.layer2(x)
        
        return x_in + x
    


class Upsampler(torch.nn.Module):
    
    def __init__(self, channels, kernel_size):
        super().__init__()
        
        self.shuffle = PixelShuffle3D(2)
        
        self.conv_layer = nn.Conv3d(in_channels  = channels, 
                                    out_channels = 8 * channels, 
                                    kernel_size  = kernel_size,
                                    stride       = 1,
                                    padding      = (kernel_size-1)//2,
                                    dtype=torch.float32)
        
    def forward(self, x):
        
        x = self.conv_layer(x)
        x = self.shuffle(x)
        
        return x
    
    
class PixelShuffle3D(nn.Module):
    
    def __init__(self, upscale_factor):
        
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        
        batched = len(x.shape) == 5
        if not batched:  # Handle unbatched data
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, channels, Nx, Ny, Nz = x.size()
        r = self.upscale_factor

        # Reshape the tensor
        x = x.view(batch_size, channels // (r**3), r, r, r, Nx, Ny, Nz)

        # Permute and rearrange the dimensions
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels // (r**3), Nx * r, Ny * r, Nz * r)

        return x