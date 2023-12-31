#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

from .subgridmodel import check_starForming
from scipy.ndimage import zoom


def classify_dataset(dataset):
    classification = []
    for i in range(len(dataset)):
        tensor = dataset[i]
        classification.append(check_starForming(tensor))

    return torch.tensor(classification)



def batch_classified_data(classified_dataset, batch_size, pytorch_hijinks=False):
    """
    Takes in a classified dataset and makes a new list which contains batches of data.

     Variables
    classified_datset: a classified datset which is a list with two entries. The first entry is
    a list of tensors for the network to be trained on. The second entry a list of classifications 
    corresponding in entry number to the previous list of tensors.
    batch_size: desired size of batches

    """
    batched_list = []
    data = []
    classification = []

    if pytorch_hijinks:
        for i in range(len(classified_dataset[1])):
            
            tensor_shape = list(torch.squeeze(classified_dataset[0][i]).shape)
            class_num_shape = list(torch.squeeze(classified_dataset[1][i]).shape)

            tensor = torch.reshape(classified_dataset[0][i], (1,*tensor_shape))
            class_num = torch.reshape(classified_dataset[0][i], (1, *class_num_shape))

            data.append(tensor)
            classification.append(class_num)

            if (i+1) % batch_size == 0:
                data = torch.stack(data)
                classification = torch.stack(classification)
                batched_list.append((data, classification))
                
                data = []
                classification = []
            
        return batched_list

    else:
        for i in range(len(classified_dataset[1])):
            
            data.append(classified_dataset[0][i])
            classification.append(classified_dataset[1][i])

            if (i+1) % batch_size == 0:
                data = torch.stack(data)
                classification = torch.stack(classification)
                batched_list.append((data, classification))
                
                data = []
                classification = []
            
        return batched_list


def star_forming_ratio(classified_dataset):
    """
    Calculates and returns ratio of star forming tensor to non star forming tensors in a classified dataset

     Variables
    classified_datset: a classified datset which is a list with two entries. 
    The first entry a list of tensors for the network to be trained on. The second 
    entry a list of classifications corresponding in entry number to the previous list of tensors.
    """
    calssification = classified_dataset[1]
    size = len(calssification)
    total_starforming = 0
    for i in range(size):
        total_starforming += calssification[i]
    return round(float(total_starforming / size), 2)



def tensor_slicer_3d(tensor, output_lenght, stride=16):
    """
    Slices given tensor into smaller volumes of desired side lenght

     Variables
    tensor: input tensor
    output_lenght: desired side lenght for slices of tensor

    return: list of slices of tensor
    """

    matrices, x, y, z = tensor.shape
    slices = [] 

    """def sigma_sum(start, end, expression):
        return sum(expression(i) for i in range(start, end))

    def dist(i):
        return ((x - i*stride) - output_lenght)//stride

    test = sigma_sum(0, output_lenght, dist)
    print(test) """

    for i in range(x//output_lenght):
        for j in range(y//output_lenght):
            for k in range(z//output_lenght):
                x_start = i * output_lenght
                y_start = j * output_lenght
                z_start = k * output_lenght
                tensor_slice = tensor[:, x_start : x_start + output_lenght,
                                         y_start : y_start + output_lenght,
                                         z_start : z_start + output_lenght]
                print(tensor_slice.shape)
                tensor_slice = torch.reshape(tensor_slice, (1, matrices, output_lenght, output_lenght, output_lenght))
                slices.append(tensor_slice)
    
    return slices



def tensor_slicer_2d(tensor, output_lenght):
    """
    Slices given tensor into smaller volumes of desired side lenght

     Variables
    tensor: input tensor
    output_lenght: desired side lenght for slices of tensor

    return: list of slices of tensor
    """

    matrices, x, y = tensor.shape
    slices = []

    for i in range(x//output_lenght):
        for j in range(y//output_lenght):
                x_start = i * output_lenght
                y_start = j * output_lenght
                tensor_slice = tensor[:, x_start : x_start + output_lenght,
                                         y_start : y_start + output_lenght]
                tensor_slice = torch.reshape(tensor_slice, (matrices, output_lenght, output_lenght))
                slices.append(tensor_slice)
    
    return slices



def classified_data_slicer(classified_data, output_lenght):
    """
    Creates a new dataset which replaces tensors with a list of slices from tensor in the input dataset. The slices are cubes.

     Variables
    classified_data: the input in which we will replace tensors with a list of slices from the tensor
    output_lenght: side lenght of the slices
    """
    sliced_data = []
    for i, tensor in enumerate(classified_data[0]):
        sliced_tensor = tensor_slicer_3d(tensor, output_lenght)
        sliced_data.append((sliced_tensor, classified_data[1][i]))
    return sliced_data



def sr_data_slicer(tensor_list, output_lenght, tensor_slicer=tensor_slicer_2d, add_dim=False):
    """
    Creates a new dataset which takes in a list of tensors with a list of slices from each tensor

     Variables
    tensor_list: list of tensors to be sliced
    output_length: lenght of the sides of the slices
    """
    sliced_tensors = []
    for i, tensor in enumerate(tensor_list):
        if add_dim:
            tensor = torch.stack([tensor])
        sliced_tensor = tensor_slicer(tensor, output_lenght)
        sliced_tensors.extend(sliced_tensor)

    return sliced_tensors

def rescale_tensors(tensors, scale_factor):
    """
    Takes a list of tensors, scales them down and back up to their original size and returns a list of those scaled tensors
    
    tensors: list of tensors
    scale_factor: number to scale the tensors
    """
    transformed_tensors = []

    for i, tensor in enumerate(tensors):

        percentage = round(100 * (i)/(len(tensors)), 1)
        print(f"{percentage}%", end="\r")

        tensor = torch.squeeze(tensor)
        
        tensor = zoom(tensor, 1/scale_factor) 
        tensor = zoom(tensor, scale_factor)
        tensor = torch.from_numpy(tensor)
        transformed_tensors.append(tensor)

    return torch.stack(transformed_tensors)