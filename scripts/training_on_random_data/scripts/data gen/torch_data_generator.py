import torch
import time
import sys
sys.path.append("C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data")
from src import subgridmodel as sgm

def gen_save_data(name, size, box_lenght):
    print("Generating Datasets")
    time_start = time.time()

    data = sgm.gen_fast_classified_data(size, box_lenght)
    torch.save(data, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/torch data/fast {box_lenght} cubed/{name}.pt")

    star_forming = sgm.star_forming_ratio(data)
    data_dict = {"size": size, "box lenght": box_lenght, "ratio": star_forming}
    torch.save(data_dict, f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/training_on_random_data/random data/torch data/fast {box_lenght} cubed/{name}_info.pt")

    print(f"{name} data properties: {data_dict}")
    time_end = time.time()
    print(f"Generated in {(time_end - time_start)/60} mins") 

BL = 16
gen_save_data(name="training", size=50000, box_lenght=BL)
gen_save_data(name="validation", size=5000, box_lenght=BL)