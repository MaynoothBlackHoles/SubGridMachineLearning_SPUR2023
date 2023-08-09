"""
Script to test avarage PSNR of a chosen dataset
"""

import torch

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution/datasets"

from subgrid_physics_modelling import network_training_utils as ntu
from subgrid_physics_modelling import synthetic_data_generation as sdg

# load your dataset
dataset_name = "dataset_3_32_2.pt"
test_data = torch.load(DATA_DIR + f"/{dataset_name}")

bicubic_data = sdg.batch_classified_data(classified_dataset=test_data["training"], batch_size=1)


# testing on loaded dataset
data_SSIM = ntu.test_metric(bicubic_data, metric=ntu.eval_SSIM)
data_PSNR = ntu.test_metric(bicubic_data, metric=ntu.eval_PSNR)

print("-----------------------------")
print(data_SSIM)
print(data_PSNR)


"""
sf 4: psnr 65
sf 8: psnr 54
sf 16: psnr 44
"""