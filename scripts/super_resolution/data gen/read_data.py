import torch

import sys
import os
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/") 

DATA_DIR = top_dir + "/data/super_resolution/datasets"

dataset_name = "dataset_2_32_2.pt"
dataset = torch.load(DATA_DIR + f"/{dataset_name}")