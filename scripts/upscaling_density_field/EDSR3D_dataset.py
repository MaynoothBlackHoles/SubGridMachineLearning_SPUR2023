#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:26:32 2023

@author: john
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init
from scipy.ndimage import zoom
from torch.utils.data import Dataset



class EDSR3DDataset(Dataset):
    """
    This Dataset class will take a list of density fields (represented as 3D 
    tensors) and process it into a dataset of rescaled patches and 
    corresponding high resolution labels.
    
    It can be passed to a pytorch DataLoader object for batching/shuffling.
    """
    
    def __init__(self, regions, scale_factor = 4, patch_size = 84):
        
        regions = [torch.tensor(region).float().unsqueeze(0) 
                   for region in regions]
        
        patches            = self.create_patches(regions, patch_size)
        downscaled_patches = self.rescale(patches, 1/scale_factor)
        upscaled_patches   = self.rescale(downscaled_patches, scale_factor)
        
        self.scale_factor     = scale_factor
        self.regions          = regions
        self.data             = downscaled_patches
        self.upscaled_patches = upscaled_patches
        self.labels           = patches
        
        self.L2               = nn.MSELoss()
        self.L1               = nn.L1Loss()
        self.dl               = DensityLoss()


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
            
            for i in range(0, Nx - patch_size, 32):
                for j in range(0, Ny - patch_size, 32):
                    for k in range(0, Nz - patch_size, 32):
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
        avg_l1, avg_l2, avg_dl = 0, 0, 0
        
        for patch, patch_bicubic in zip(self.labels, self.upscaled_patches):
            loss = self.L1(patch, patch_bicubic).item()
            avg_l1 += loss
            
            loss = self.L2(patch, patch_bicubic).item()
            avg_l2 += loss
            
            loss = self.dl(patch, patch_bicubic).item()
            avg_dl += loss
            
        self.avg_l1 = avg_l1 / len(self.labels)
        self.avg_l2 = avg_l2 / len(self.labels)
        self.avg_dl = avg_dl / len(self.labels)
      

        
def expand_dataset(tensors):
    expanded_dataset = []
    
    for tensor in tensors:
        transformed_tensors = get_tensor_transformations(tensor)
        expanded_dataset += transformed_tensors
        
    return expanded_dataset


def get_tensor_transformations(tensor):
    transformed_tensors = []
    
    tesnor_xyz = tensor
    tesnor_zyx = tensor.transpose(2, 1, 0)
    tesnor_xzy = tensor.transpose(0, 2, 1)
    
    tesnor_xyZ = np.flip(tesnor_xyz, 2).copy()
    tesnor_zyX = np.flip(tesnor_zyx, 2).copy()
    tesnor_xzY = np.flip(tesnor_xzy, 2).copy()
    
    ts = [tesnor_xyz, 
          tesnor_zyx, 
          # tesnor_xzy, 
          # tesnor_xyZ, 
          # tesnor_zyX, 
          tesnor_xzY]
    
    for t in ts:
        transformed_tensors += get_tensor_plane_transformations(t)
    
    return transformed_tensors


def get_tensor_plane_transformations(t):
    transformed_tensors = []
    
    for i in range(4):
        t = t.transpose(1, 0, 2)
        transformed_tensors.append(t)
        
        t = np.flip(t, 0).copy()
        transformed_tensors.append(t)
    
    return transformed_tensors


class DensityLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.L1 = nn.L1Loss()

    def forward(self, x, y):
        
        loss = self.L1(10**x, 10**y)
        
        return loss