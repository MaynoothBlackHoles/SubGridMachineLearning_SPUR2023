import torch
import time
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import subgridmodel as sgm

def gen_save_data(name, size, box_lenght):
    print("Generating Datasets")
    time_start = time.time()

    max_stars = 160*8
    data = sgm.gen_fast_classified_data(size, box_lenght, max_stars)
    torch.save(data, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {box_lenght} cubed/{name}.pt")

    star_forming = sgm.star_forming_ratio(data)
    data_dict = {"size": size, "box lenght": box_lenght, "ratio": star_forming, "max stars": max_stars}
    torch.save(data_dict, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/fast {box_lenght} cubed/{name}_info.pt")

    print(f"{name} data properties: {data_dict}")
    time_end = time.time()
    print(f"Generated in {(time_end - time_start)/60} mins")

def gen_save_data_uniform(name, size, box_lenght):
    print("Generating Datasets")
    time_start = time.time()

    data = sgm.gen_classified_data(size, box_lenght)
    torch.save(data, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/{box_lenght} cubed/{name}.pt")

    star_forming = sgm.star_forming_ratio(data)
    data_dict = {"size": size, "box lenght": box_lenght, "ratio": star_forming}
    torch.save(data_dict, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/{box_lenght} cubed/{name}_info.pt")

    print(f"{name} data properties: {data_dict}")
    time_end = time.time()
    print(f"Generated in {(time_end - time_start)/60} mins")

BL = 4
#gen_save_data(name="training", size=200, box_lenght=BL)
#gen_save_data(name="validation", size=1000, box_lenght=BL)
gen_save_data_uniform(name="testing", size=50000, box_lenght=BL)
gen_save_data_uniform(name="validation", size=10000, box_lenght=BL)