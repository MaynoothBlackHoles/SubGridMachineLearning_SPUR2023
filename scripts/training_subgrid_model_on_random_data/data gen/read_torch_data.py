"""
Script which gives basic information on a dataset
"""

import torch

BL = 4 # box lenght

# loading info about desired datasets
train_info = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/training_info.pt")
validation_info = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {BL} cubed/validation_info.pt")

print(train_info)
print(validation_info)