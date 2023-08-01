import torch
import random
import numpy as np

import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)

from data_utils import classify_dataset, batch_classified_data
from subgridmodel import H2_crit, rho_crit_star, rho_crit_BH


def gen_uniform_tensor(box_lenght):
    """
    Generates a tensor with uniformly disrtibued properties

     Variables
    box_lenght: side lenght of the generated cube
    """
    cool_factor = 1 / box_lenght 
    tensor_size = (box_lenght, box_lenght, box_lenght)

    number_density = torch.rand(*tensor_size) * (cool_factor + 1) * 100
    H2_fraction    = torch.rand(*tensor_size) * (cool_factor + 1) * 1e-3
    freefall_time  = torch.rand(*tensor_size)
    cooling_time   = torch.rand(*tensor_size) + (1 - (cool_factor)**(1/2))
    divergence     = torch.rand(*tensor_size) + (-cool_factor)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)



def gen_starforming_tensor(box_lenght):
    number_density = 100 + torch.rand(box_lenght, box_lenght, box_lenght)
    H2_fraction =   1e-3 + torch.rand(box_lenght, box_lenght, box_lenght) 
    freefall_time = torch.rand(box_lenght, box_lenght, box_lenght)
    cooling_time = 1 + torch.rand(box_lenght, box_lenght, box_lenght)
    divergence = -1 * torch.rand(box_lenght, box_lenght, box_lenght)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)



def gen_NONstarforming_tensor(box_lenght):
    number_density = 100 * torch.rand(box_lenght, box_lenght, box_lenght)
    H2_fraction =   1e-3 * torch.rand(box_lenght, box_lenght, box_lenght) 
    freefall_time = 1 + torch.rand(box_lenght, box_lenght, box_lenght)
    cooling_time = torch.rand(box_lenght, box_lenght, box_lenght)
    divergence = torch.rand(box_lenght, box_lenght, box_lenght)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)



def generate_log_tensor(box_lenght):
    zeros_tensor = torch.zeros(box_lenght, box_lenght, box_lenght)
    number_density = zeros_tensor.log_normal_(mean=1, std=2)
    H2_fraction = zeros_tensor.log_normal_(mean=1, std=2)
    freefall_time = zeros_tensor.log_normal_(mean=1, std=2)
    cooling_time = zeros_tensor.log_normal_(mean=1, std=2)
    divergence =  zeros_tensor.log_normal_(mean=1, std=2)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)



def generate_dataset(size, box_lenght, tensor_generator=gen_uniform_tensor):
    dataset = []
    for i in range(size):
        dataset.append(tensor_generator(box_lenght))

    return torch.stack(dataset, dim=0)



def gen_classified_data(size, box_length, tensor_generator=gen_uniform_tensor):
    data = generate_dataset(size, box_length, tensor_generator)
    classification = classify_dataset(data)
    return data, classification



def gen_batched_classified_data(size, box_length, batch_size, tensor_generator=gen_uniform_tensor):
    data = generate_dataset(size, box_length, tensor_generator)
    classification = classify_dataset(data)
    classified_data = (data, classification)
    batched_data = batch_classified_data(classified_data, batch_size)
    return batched_data



def gen_fast_classified_data(size, box_lenght, max_stars=5, min_stars=1):
    """
    Creates a dataset with uniformly distributed properties by generating a non star forming tensor,
    and randomly deciding whether to make it a star forming tensor by randomly inserting star forming
    regions. Significantly faster generation for large datasets with large tensors.

     Variables
    size: desired amount of tensors in dataset
    boz_lenght: length of the sides of the "cube" to be generated

    return: tuple of list of tensors and list of classification
    """
    data = []
    classification = []

    for i in range(size):

        # progress tracker
        if (i + 1) % 100 == 0:
            percentage = round(100 * ((i + 1)/size), 1)
            print(f" {percentage}% done", end="\r")

        # generates a non star forming tensor
        tensor = gen_NONstarforming_tensor(box_lenght)
        
        # 50% chance to decide on star forming
        star_forming = random.randint(0,1)
        if star_forming:

            num_stars = random.randint(min_stars, max_stars)

            # creates star forming voxel properties and replaces existing 
            # properties in tensor
            for i in range(num_stars):
                x = random.randint(0, box_lenght - 1)
                y = random.randint(0, box_lenght - 1)
                z = random.randint(0, box_lenght - 1)

                nd   = random.uniform(100, 110)
                H2_f = random.uniform(1e-3, .1 * 1e-3)
                fft  = random.uniform(0.9, 1)
                ct   = random.uniform(1, 0.1)
                div  = random.uniform(-0.1, 0)

                tensor[0, x, y, z] = nd
                tensor[1, x, y, z] = H2_f
                tensor[2, x, y, z] = fft
                tensor[3, x, y, z] = ct
                tensor[4, x, y, z] = div

        # adding tensor to list and classification to list
        data.append(tensor)
        classification.append(torch.tensor(star_forming))

    # converts both lists into pytorch tensors
    data = torch.stack(data, dim=0)
    classification = torch.stack(classification, dim=0)

    return (data, classification)


###############################################################################
#        Generating Data for Subgrid Model with Stars and Black Holes         #
###############################################################################


def generate_random_compact_object_data(size, box_length):
    """
    This function returns a dataset consisting of randomly generated tensors, 
    representing regions, which are classified into four classes:
        0 - an empty region
        1 - a star forming region
        2 - a black hole forming region
        3 - a region with both stars and black holes
    """
    data, labels = [], []
    min_objects = 1
    max_objects = 5
    
    for i in range(size):
        
        X = generate_empty_region(box_length)
        label = random.randint(0, 3)
        num_objects = random.randint(min_objects, max_objects)
        
        if label == 1:
            add_compact_objects(X, num_objects * [rho_crit_star])
            
        elif label == 2:
            add_compact_objects(X, num_objects * [rho_crit_BH])
            
        elif label == 3:
            n_star = (num_objects // 2) + 1
            n_BH   = (num_objects // 2) + 1
            rhos = n_star * [rho_crit_star] + n_BH * [rho_crit_BH]
            add_compact_objects(X, rhos)
        
        data.append(X)
        labels.append(torch.tensor(label))
        
    return torch.stack(data, dim=0), torch.stack(labels, dim=0)



def generate_empty_region(box_length):
    """
    Returns a tensor representing an empty region. The tensor is generated
    randomly by randomly selecting at least one condition for compact object
    formation to break.
    """
    # Create tensor with random entries for velocity components
    X = torch.randn(6, box_length, box_length, box_length)
    
    # Iterate over all sites of the tensor.
    for i in range(box_length):
        for j in range(box_length):
            for k in range(box_length):
    
                # Radnomly choose at least 1 condition for compact object
                # formation to break.
                conditions_broken = random.choices([True, False], k = 3)
                while sum(conditions_broken) == 0:
                    conditions_broken = random.choices([True, False], k = 3)
                    
                # Add a velocity divergence
                if conditions_broken[0]:
                    X[5, i, j, k] = torch.rand(1)
                else:
                    X[5, i, j, k] = -torch.rand(1)
                    
                # Add a H2 fraction
                if conditions_broken[1]:
                    X[1, i, j, k] = H2_crit * torch.rand(1)
                else:
                    X[1, i, j, k] = H2_crit * (torch.rand(1) + 1)
                    
                # Add a density
                if conditions_broken[2]:
                    X[0, i, j, k] = rho_crit_star * torch.rand(1)
                else:
                    X[0, i, j, k] = rho_crit_star * (torch.rand(1) + 1)
        
    return X
    


def add_compact_objects(X, rhos):
    """
    This function randomly select n distinct sites of region X and adds compact
    objects to them, where n is the length of the given list of rho values. The
    compact object added to a site is determined by the corresponding rho
    value. 
    """
    region_size = X.numel()
    num_objects = len(rhos)
    
    if num_objects > region_size:
        raise ValueError("Number of objects must be less than region size")

    sites = np.random.choice(region_size, size=num_objects, replace=False)
    site_inds = np.unravel_index(sites, X.shape)
    
    for rho, i, j, k in zip(rhos, site_inds[1], site_inds[2], site_inds[3]):
        X[5, i, j, k] = -torch.rand(1)
        X[1, i, j, k] = (torch.rand(1) + 1) * H2_crit
        X[0, i, j, k] = (torch.rand(1) + 1) * rho