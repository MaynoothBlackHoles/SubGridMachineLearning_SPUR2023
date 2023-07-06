import torch


def check_pixel(tensor, x, y, z):

    number_density = tensor[0, x, y, z]
    H2_fraction = tensor[1, x , y, z]
    freefall_time = tensor[2, x, y, z]
    cooling_time = tensor[3, x, y, z]
    divergence = tensor[4, x, y, z]
    
    if number_density >= 100 and H2_fraction >= 10**-3 and freefall_time < cooling_time and divergence < 0:
        return True
    else:
        return False

def check_tensor(tensor):
    star_forming_pixels = []

    matrices, x, y, z = tensor.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if check_pixel(tensor, i, j, k):
                    star_forming_pixels.append([i, j, k])
    
    return star_forming_pixels

def check_starForming(tensor):
    if check_tensor(tensor):
        return 1 # star forming tensor
    else:
        return 0 # non star forming tensor
    
def generate_tensor(box_lenght, max_nd=1.1*100, max_H2frac=1.1*10**-3):
    number_density = max_nd * torch.rand(box_lenght, box_lenght, box_lenght) # maybe get max and min values for all
    H2_fraction = max_H2frac * torch.rand(box_lenght, box_lenght, box_lenght)
    freefall_time = torch.rand(box_lenght, box_lenght, box_lenght)
    cooling_time = torch.rand(box_lenght, box_lenght, box_lenght)
    divergence = -0.5 + torch.rand(box_lenght, box_lenght, box_lenght)

    tensor = [number_density,
              H2_fraction,
              freefall_time,
              cooling_time,
              divergence]

    return torch.stack(tensor, dim=0)

def generate_dataset(size, box_lenght):
    dataset = []
    for i in range(size):
        dataset.append(generate_tensor(box_lenght))

    return torch.stack(dataset, dim=0)

def classify_dataset(dataset):
    classification = []
    for i in range(len(dataset)):
        tensor = dataset[i]
        classification.append(check_starForming(tensor))

    return torch.tensor(classification)

def gen_classified_data(size, box_length):
    data = generate_dataset(size, box_length)
    classification = classify_dataset(data)
    return data, classification

def gen_batched_classified_data(size, box_lenght, batch_size):
    batched_list = []
    for i in range(size // batch_size):
        data = generate_dataset(batch_size, box_lenght)
        classification = classify_dataset(data)
        batched_list.append((data, classification))

    return batched_list

def batch_classified_data(classified_dataset, batch_size):
    batched_list = []
    batch = []
    
    for i in range(len(classified_dataset)):
        data = []
        data.append(classified_dataset[i][0])
        batch.append(data)

        classification = []
        classification.append(classified_dataset[i][0])
        batch.append(classification)

        if i + 1 % batch_size == 0:
            batched_list.append(batch)
            batch = []


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
print("test data done")
"""


cool_data = gen_classified_data(100, 32) # ran one got all ones
print(cool_data[1])

#problem: need to know roughly how likely a star will form in a tensor, or how to tweek random numbers paramaters 
# to make this random data look like real data, i could also just tweak paramaters myslef..









