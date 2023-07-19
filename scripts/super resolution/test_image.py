"""

"""

import torch
from torch import nn
import torchvision
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src import network_function as nf
from src import sr_networks as net
from src import subgridmodel as sgm

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net.Srcnn().to(device)
model.load_state_dict(torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/srcnn100.pt"))
loss_fn = nn.MSELoss()

print("[INFO] Loading datasets")
test_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/super resolution/data/training_100s.pt")

image_num = 1

low_res_tensor = test_data[0][image_num]
high_res_tensor = test_data[1][image_num]
srcnn_tensor = model(low_res_tensor)

transform = torchvision.transforms.ToPILImage()

img = transform(low_res_tensor)
img.show()
img = transform(high_res_tensor)
img.show()
img = transform(srcnn_tensor)
img.show()




print("[INFO] Done! :D")