#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.v2 as transforms

from .subgridmodel import check_starForming
from scipy.ndimage import zoom


def classify_dataset(dataset):
    classification = []
    for i in range(len(dataset)):
        tensor = dataset[i]
        classification.append(check_starForming(tensor))

    return torch.tensor(classification)



def batch_classified_data(classified_dataset, batch_size):
    """
    Takes in a classified dataset and makes a new list which contains batches of data.

     Variables
    classified_datset: a classified datset which is a list with two entries. The first entry a list of tensors for the network to be trained on. The second entry a list of classifications corresponding in entry number to the previous list of tensors.
    batch_size: desired size of batches
    """
    batched_list = []
    data = []
    classification = []

    print(len(classified_dataset))

    for i in range(len(classified_dataset[1])):
        
        # added this bit for super res issue but may mess up previus stuff
        #######################################################################
        tensor_shape = list(torch.squeeze(classified_dataset[0][i]).shape)
        class_num_shape = list(torch.squeeze(classified_dataset[1][i]).shape)

        tensor = torch.reshape(classified_dataset[0][i], (1,*tensor_shape))
        class_num = torch.reshape(classified_dataset[0][i], (1, *class_num_shape))

        data.append(tensor)
        classification.append(class_num)
        ########################################################################

        # un comment this and comment out the above if having issues
        #data.append(classified_dataset[0][i])
        #classification.append(classified_dataset[0][i])

        if (i+1) % batch_size == 0:
            data = torch.stack(data)
            classification = torch.stack(classification)
            print(data.shape)
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



def tensor_slicer_3d(tensor, output_lenght):
    """
    Slices given tensor into smaller volumes of desired side lenght

     Variables
    tensor: input tensor
    output_lenght: desired side lenght for slices of tensor

    return: list of slices of tensor
    """

    matrices, x, y, z = tensor.shape
    slices = []

    for i in range(x//output_lenght):
        for j in range(y//output_lenght):
            for k in range(z//output_lenght):
                x_start = i * output_lenght
                y_start = j * output_lenght
                z_start = k * output_lenght
                tensor_slice = tensor[:, x_start : x_start + output_lenght,
                                         y_start : y_start + output_lenght,
                                         z_start : z_start + output_lenght]
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



def sr_data_slicer(tensor_list, output_lenght, tensor_slicer=tensor_slicer_2d):
    """
    Creates a new dataset which takes in a list of tensors with a list of slices from each tensor

     Variables
    tensor_list: list of tensors to be sliced
    output_length: lenght of the sides of the slices
    """
    sliced_tensors = []
    for i, tensor in enumerate(tensor_list):
        sliced_tensor = tensor_slicer(tensor, output_lenght)
        sliced_tensors.extend(sliced_tensor)

    return sliced_tensors



def transform_tensors(tensors):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = torch.from_numpy(tensor)
        transformed_tensors.append(tensor)
    return torch.stack(transformed_tensors)



def downscale_tensors(tensors, scale_factor):
    transformed_tensors = []

    for i, tensor in enumerate(tensors):

        percentage = round(100 * (i)/(len(tensors)), 1)
        print(f"{percentage}%", end="\r")

        tensor = torch.squeeze(tensor)
        dim = len(tensor.shape)
        
        scale_down = [1/scale_factor] * (dim - 1)
        scale_up = [scale_factor] * (dim - 1)
        
        tensor = zoom(tensor, (1, *scale_down))
        tensor = zoom(tensor, (1, *scale_up))
        tensor = torch.from_numpy(tensor)
        transformed_tensors.append(tensor)

    return torch.stack(transformed_tensors)