"""
Script to hold data generating, classifying, pre processing and testing functions
"""

import torch
import random
import torchvision.transforms.v2 as transforms


def check_voxel(tensor, x, y, z):
    """
    Checks a if a given coorindate (x, y, z) of a tensor is star forming or not

    return: True or False
    """

    number_density = tensor[0, x, y, z]
    H2_fraction = tensor[1, x , y, z]
    freefall_time = tensor[2, x, y, z]
    cooling_time = tensor[3, x, y, z]
    divergence = tensor[4, x, y, z]
    
    if (number_density >= 100) and (H2_fraction >= 1e-3) and (freefall_time < cooling_time) and (divergence < 0):
        return True
    else:
        return False

def check_tensor(tensor, Full_list=False):
    """
    Checks whether a tensor if star forming or not by checking if any entry is star forming.
    
     Variables
    tensor: input tensor
    Full_list: If this is set to be True then this function will output each entry of the tensor which is star forming in a list

    return: True or False
    """

    matrices, x, y, z = tensor.shape
    star_forming_pixels = []
    
    if Full_list:
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_voxel(tensor, i, j, k):
                        star_forming_pixels.append([i, j, k])

        return star_forming_pixels
    
    else:
        break_state = False
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_voxel(tensor, i, j, k):
                        break_state = True
                        break
                if break_state:
                    break
            if break_state:
                break
        
        return break_state

def check_starForming(tensor):
    if check_tensor(tensor):
        return 1 # star forming tensor
    else:
        return 0 # non star forming tensor
    

def gen_uniform_tensor(box_lenght):
    """
    Generates a tensor with uniformly disrtibued properties

     Variables
    box_lenght: side lenght of the generated cube
    """
    cool_factor = 1 / box_lenght 
    ones_tensor = torch.ones(box_lenght, box_lenght, box_lenght)

    number_density = (cool_factor + 1) * 100 * torch.rand(box_lenght, box_lenght, box_lenght) 
    H2_fraction =  (cool_factor + 1) * 1e-3 * torch.rand(box_lenght, box_lenght, box_lenght) 
    freefall_time = torch.rand(box_lenght, box_lenght, box_lenght)
    cooling_time = ( 1 - (cool_factor)**(1/2)) * ones_tensor + torch.rand(box_lenght, box_lenght, box_lenght)
    divergence = ( - cool_factor) * ones_tensor + torch.rand(box_lenght, box_lenght, box_lenght)

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

            # creates star forming voxel properties and replaces existing properties in tensor
            for i in range(num_stars):
                x = random.randint(0, box_lenght - 1)
                y = random.randint(0, box_lenght - 1)
                z = random.randint(0, box_lenght - 1)

                nd = random.uniform(100, 110)
                H2_f = random.uniform(1e-3, .1 * 1e-3)
                fft = random.uniform(0.9, 1)
                ct = random.uniform(1, 0.1)
                div = random.uniform(-0.1, 0)

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

def sr_data_slicer(tensor_list, output_lenght):
    """
    Creates a new dataset which takes in a list of tensors with a list of slices from each tensor

     Variables
    tensor_list: list of tensors to be sliced
    output_length: lenght of the sides of the slices
    """
    sliced_tensors = []
    for i, tensor in enumerate(tensor_list):
        sliced_tensor = tensor_slicer_2d(tensor, output_lenght)
        sliced_tensors.extend(sliced_tensor)

    return sliced_tensors

def transform_tensors(tensors, transform=transforms.ToTensor()):
    transformed_tensors = []
    for i, tensor in enumerate(tensors):
        tensor = transform(tensor)
        transformed_tensors.append(tensor)
    return transformed_tensors