"""
Script used to generate datasets
"""

import torch
import time

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/random_data"

from subgrid_physics_modelling import data_utils
from subgrid_physics_modelling import synthetic_data_generation as sdg

DATA_DIR = top_dir + "/data/random_data"


def gen_save_data(name, size, box_lenght, data_dir = DATA_DIR):
    """
    Generates and saves a dataset

    parameters:
        name:       Choose what you want your generated dataset to be called,
                    this should be a string
                    
        size:       desired amount of tensors in dataset
        
        box_lenght: length of the sides of the "cube" to be generated
    """
    print("Generating Datasets")
    time_start = time.time()
    dir_name = data_dir + f"/datasets"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # creating the dataset and save the dataset to desired location
    max_stars = 20
    data = sdg.gen_fast_classified_data(size, box_lenght, max_stars)
    torch.save(data, dir_name + f"/{name}.pt")

    # creating a dictinoatry for general information of the created dataset
    star_forming = data_utils.star_forming_ratio(data)
    data_dict = {"size": size, 
                 "box lenght": box_lenght, 
                 "ratio": star_forming,
                 "max stars": max_stars}
    
    # saving the dictionary to desired location
    torch.save(data_dict, dir_name + f"/{name}_info.pt")
    print(f"{name} data properties: {data_dict}")
    time_end = time.time()
    print(f"Generated in {(time_end - time_start)/60} mins")


BL = 8 # Box Lenght
gen_save_data(name=f"training_{BL}", size=20000, box_lenght=BL)
gen_save_data(name=f"validation_{BL}", size=1000, box_lenght=BL)