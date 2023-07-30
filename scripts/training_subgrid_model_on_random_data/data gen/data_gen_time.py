"""
Script used test how long it takes to generate a dataset
"""

import os
import sys
import time

current_dir = os.getcwd()
sys.path.append(current_dir + "/../../..")

from subgrid_physics_modelling import synthetic_data_generation as sdg

time_start = time.time()
data = sdg.generate_dataset(size=1, box_lenght=64)
time_end = time.time()
print(f"Time to gen data: {time_end - time_start}")