import torch
import random

def check_pixel(tensor, x, y, z):

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
    star_forming_pixels = []

    matrices, x, y, z = tensor.shape
    
    if Full_list:
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_pixel(tensor, i, j, k):
                        star_forming_pixels.append([i, j, k])

        return star_forming_pixels
    
    else:
        break_state = False
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_pixel(tensor, i, j, k):
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
    # maybe get max and min values for all

    cool_factor = 1 / box_lenght #( 1 / (2 * (box_lenght**3)) )**(1/5)
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


def gen_fast_classified_data(size, box_lenght, max_stars=5):
    data = []
    classification = []

    for i in range(size):
        if (i + 1) % 100 == 0:
            percentage = round(100 * ((i + 1)/size), 1)
            print(f" {percentage}% done", end="\r")

        # generates a non star forming tensor
        tensor = gen_NONstarforming_tensor(box_lenght)
        
        # filps a coin, and puts in some stars if flipped right
        star_forming = random.randint(0,1)
        if star_forming:

            num_stars = random.randint(1, max_stars)

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

        data.append(tensor)
        classification.append(torch.tensor(star_forming))

    data = torch.stack(data, dim=0)
    classification = torch.stack(classification, dim=0)
    return (data, classification)
    

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
    batched_list = []
    data = []
    classification = []

    for i in range(len(classified_dataset[1])):
        data.append(classified_dataset[0][i])
        classification.append(classified_dataset[1][i])

        # this fucker i needed to have brackets here.....
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


def star_forming_ratio(classified_dataset):
    calssification = classified_dataset[1]
    size = len(calssification)
    total_starforming = 0
    for i in range(size):
        total_starforming += calssification[i]
    return round(float(total_starforming / size), 2)
