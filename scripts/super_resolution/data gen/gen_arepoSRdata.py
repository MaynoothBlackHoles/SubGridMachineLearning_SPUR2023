import torch
import random
import numpy as np

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution/datasets"

from subgrid_physics_modelling import data_utils as du

# parameters
SCALE_FACTOR = 8
IMAGE_SLICE_SIZE = 32 # keep the scale factor as a multiple of the scale factor
BIG_TENSORS = 1 # max = 125
CHANNEL_NUM = 0
SCALE_DATA = 1e+28

print("[INFO] Loading data")
tensors_dict = np.load(DATA_DIR + "/snap_007_tensors.npz")

tensors_list = []
tick = 0
for key in tensors_dict:
    tensors_list.append(torch.tensor(tensors_dict[key]).permute(3, 0, 1, 2)[CHANNEL_NUM] * SCALE_DATA)

    percentage = round(100 * (tick)/BIG_TENSORS, 1)
    print(f"{percentage}%", end="\r")

    tick += 1
    if tick == BIG_TENSORS:
        break


print("[INFO] Creating datasets")
sliced_tensor_list = du.sr_data_slicer(tensors_list, IMAGE_SLICE_SIZE, tensor_slicer=du.tensor_slicer_3d, add_dim=True)
random.shuffle(sliced_tensor_list)
 
print("[INFO] Transforming tensors")
downscaled = du.rescale_tensors(sliced_tensor_list, scale_factor=SCALE_FACTOR)
high_res = torch.stack(sliced_tensor_list)

split_num = int(len(sliced_tensor_list) * 0.9)
print(f"[INFO] Total amount of samples: {len(sliced_tensor_list)}")

training = (downscaled[:split_num], high_res[:split_num])
validation = (downscaled[split_num:], high_res[split_num:])

# saving data
print("[INFO] Saving datasets")
data_dict = {"training": training,
              "validation": validation,
              "properties": {"image patch size": IMAGE_SLICE_SIZE, "dataset size": len(sliced_tensor_list), "scale factor": SCALE_FACTOR}}

def save_data(dataset, name):
    torch.save(dataset, DATA_DIR + f"/{name}.pt")

save_data(data_dict, name=f"dataset_{BIG_TENSORS}_{IMAGE_SLICE_SIZE}_{SCALE_FACTOR}", )

print("[INFO] Done!")