import torch

import sys
import os
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/") # this line is here for windows, if on linux this does nothing

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/") 
sys.path.append(parent_dir)

dataset_name = input("[INPUT] Name of dataset: ")
dataset = torch.load(current_dir + f"/data/{dataset_name}")

print(dataset["properties"])