"""
Script used to generate datasets
"""

import torch
import time
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm

def gen_save_data(name, size, box_lenght):
    """
    Generates and saves a dataset

     Variables
    name: Choose what you want your generated dataset to be called, this should be a string
    size: desired amount of tensors in dataset
    box_lenght: length of the sides of the "cube" to be generated
    """
    print("Generating Datasets")
    time_start = time.time()

    # creating the dataset
    max_stars = 5
    data = sgm.gen_fast_classified_data(size, box_lenght, max_stars)
    # save the dataset to desired location
    torch.save(data, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {box_lenght} cubed/{name}.pt")

    # creating a dictinoatry for general information of the created dataset above
    star_forming = sgm.star_forming_ratio(data)
    data_dict = {"size": size, "box lenght": box_lenght, "ratio": star_forming, "max stars": max_stars}
    # saving the dictionary to desired location
    torch.save(data_dict, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {box_lenght} cubed/{name}_info.pt")

    print(f"{name} data properties: {data_dict}")
    time_end = time.time()
    print(f"Generated in {(time_end - time_start)/60} mins")


BL = 4 # Box Lenght
gen_save_data(name="training", size=200, box_lenght=BL)
gen_save_data(name="validation", size=1000, box_lenght=BL)