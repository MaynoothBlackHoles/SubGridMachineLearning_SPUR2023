#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:36:18 2023

@author: john

This file defines classes used to create pytorch models and datasets for
training a model to upscale a density field.
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class SRCNN3(torch.nn.Module):
    """
    This class defines a basic SRCNN network with a residual connection for
    upsampling a 3D density field
    """
    
    def __init__(self, f_1 = 9, f_2 = 1, f_3 = 5, n1 = 64, n2 = 32):
        
        super().__init__()
        
        self.layer1 = nn.Conv3d(in_channels  = 1, 
                                out_channels = n1, 
                                kernel_size  = f_1, 
                                stride       = 1,
                                padding      = 4,
                                dtype=torch.float32)
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Conv3d(in_channels  = n1,
                                out_channels = n2,
                                kernel_size  = f_2,
                                stride       = 1,
                                dtype=torch.float32)
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Conv3d(in_channels  = n2,
                                out_channels = 1,
                                kernel_size  = f_3,
                                stride       = 1,
                                padding      = 2,
                                dtype=torch.float32)
        
    def forward(self, x_in):
        
        x = self.layer1(x_in)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        
        return x_in + x
    
    
    
class SRDataset3(Dataset):
    """
    This Dataset class will take a list of density fields (represented as 3D 
    tensors) and process it into a dataset of rescaled patches and 
    corresponding high resolution labels.
    
    It can be passed to a pytorch DataLoader object for batching/shuffling.
    """
    
    def __init__(self, regions, scale_factor = 4, patch_size = 32, crop = 6):
        
        regions = [torch.tensor(region).float().unsqueeze(0) 
                   for region in regions]
        
        patches            = self.create_patches(regions, patch_size)
        downscaled_patches = self.rescale(patches, 1/scale_factor)
        upscaled_patches   = self.rescale(downscaled_patches, scale_factor)
        
        # For regular SRCNN we might want to cropt the labels as SRCNN output
        # is smaller than its input.
        if crop:
            labels = [y[:, crop:-crop, crop:-crop, crop:-crop] 
                      for y in patches]
        else:
            labels = patches
        
        self.scale_factor       = scale_factor
        self.regions            = regions
        self.patches            = patches
        self.downscaled_patches = downscaled_patches
        self.data               = upscaled_patches
        self.labels             = labels
        self.criterion          = nn.MSELoss()


    def __len__(self):
        """
        This function is required by the DataLoader interface
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        This function is required by the DataLoader interface
        """
        return self.data[idx], self.labels[idx]
    
    
    def create_patches(self, regions, patch_size):
        """
        Create a set of patches from the given set of regions
        """
        patches = []
        
        for region in regions:
            Nc, Nx, Ny, Nz = region.shape
            
            for i in range(0, Nx, patch_size):
                for j in range(0, Ny, patch_size):
                    for k in range(0, Nz, patch_size):
                        if i+patch_size < Nx:
                            if j+patch_size < Ny:
                                if k+patch_size < Nz:
                                    patch = region[:,
                                                   i:i + patch_size, 
                                                   j:j + patch_size,
                                                   k:k + patch_size]
                                    patches.append(patch)
                            
        return patches


    def rescale(self, regions, scale_factor):
        """
        Rescale the given regions by the given scale factor.
        """
        rescaled_regions = []
        for region in regions:
            rescaled_region = zoom(region[0, :, :, :],
                                   scale_factor, 
                                   order=3)
            rescaled_regions.append(torch.tensor(rescaled_region).unsqueeze(0))
            
        return rescaled_regions
    
    
    def compute_bicubic_metrics(self):
        """
        Compute the average loss of the bicubic upscaling as a baseline 
        measurement.
        """
        avg_loss = 0
        
        for patch, patch_bicubic in zip(self.patches, self.data):
            loss = self.criterion(patch, patch_bicubic).item()
            avg_loss += loss
            
        self.avg_loss = avg_loss / len(self.patches)