import torch
import time


def check_pixel(tensor, x, y, z):

    number_density = tensor[0, x, y, z]
    H2_fraction = tensor[1, x , y, z]
    freefall_time = tensor[2, x, y, z]
    cooling_time = tensor[3, x, y, z]
    divergence = tensor[4, x, y, z]
    
    if number_density >= 100 and H2_fraction >= 1e-3 and freefall_time < cooling_time and divergence < 0:
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
        
        if break_state:
            return True
        else: 
            return False

def check_starForming(tensor):
    if check_tensor(tensor):
        return 1 # star forming tensor
    else:
        return 0 # non star forming tensor
    

def generate_tensor(box_lenght):
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
    divergence = 1 * torch.rand(box_lenght, box_lenght, box_lenght)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)

def generate_log_tensor(box_lenght):
    number_density = torch.zeros(box_lenght, box_lenght, box_lenght).log_normal_(mean=1, std=2)
    H2_fraction = torch.zeros(box_lenght, box_lenght, box_lenght).log_normal_(mean=1, std=2)
    freefall_time = torch.zeros(box_lenght, box_lenght, box_lenght).log_normal_(mean=1, std=2)
    cooling_time = torch.zeros(box_lenght, box_lenght, box_lenght).log_normal_(mean=1, std=2)
    divergence =  torch.zeros(box_lenght, box_lenght, box_lenght).log_normal_(mean=1, std=2)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)


def generate_dataset(size, box_lenght, tensor_generator=generate_tensor): # add option to easily change tensor generator
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

def gen_classified_data(size, box_length, tensor_generator=generate_tensor):
    data = generate_dataset(size, box_length, tensor_generator)
    classification = classify_dataset(data)
    return data, classification

def gen_batched_classified_data(size, box_lenght, batch_size, tensor_generator=generate_tensor):
    batched_list = []
    for i in range(size // batch_size):
        data = generate_dataset(batch_size, box_lenght, tensor_generator)
        classification = classify_dataset(data)
        batched_list.append((data, classification))

    return batched_list

def batch_classified_data(classified_dataset, batch_size):
    batched_list = []
    data = []
    classification = []

    for i in range(len(classified_dataset[1])):
        print(batched_list)
        #print(i)

        data.append(classified_dataset[0][i])
        #print(classified_dataset[0][i])
        classification.append(classified_dataset[1][i])
        #print(int(classified_dataset[1][i]))

        #print(i)
        if i + 1 % batch_size == 0:
            #print([data, classification])
            data = torch.stack(data)
            classification = torch.stack(classification)
            batched_list.append((data, classification))
            batched_list.append("lol")
            #print(batched_list)

            data = []
            classification = []
        
    return batched_list



#epic_data = gen_batched_classified_data(size=4, box_lenght=2, batch_size=2)

# code to put generated data into a file to pick out later

"""import shelve
data = shelve.open('batched_data.db')
LENGTH = 16
BATCH_SIZE = 64
data['training'] = gen_batched_classified_data(size=100, box_lenght=LENGTH, batch_size=BATCH_SIZE)
print("trining data done")
data['validation'] = gen_batched_classified_data(size=50, box_lenght=LENGTH, batch_size=BATCH_SIZE)
print("validataion data done")
data['test'] = gen_batched_classified_data(size=50, box_lenght=LENGTH, batch_size=BATCH_SIZE)
print("test data done")"""


start_time = time.time()
BL = 16
print(f"Box Size: {BL}^3")
start_time = time.time()
cool_data = gen_classified_data(size=100, box_length=BL)
end_time = time.time()
print(f"Time taken to generate classified dataset: {end_time - start_time}")
print(cool_data[1])

"""start_time = time.time()
batched_data = batch_classified_data(cool_data, batch_size=5)
end_time = time.time()
print(f"Time taken to batch: {end_time - start_time}")
print(batched_data)
"""
def prob(mylist):
    total = 0
    for i in range(len(mylist)):
        total += mylist[i]
    print(f"Ratio: {float(total / len(mylist)):.6f}")

prob(cool_data[1])







