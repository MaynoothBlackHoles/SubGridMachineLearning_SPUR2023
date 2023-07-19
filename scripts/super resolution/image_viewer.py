import torch
import torchvision.transforms as T

train_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/super resolution/data/training.pt")

image_num = 1

tensor_1 = train_data[0][image_num]
tensor_2 = train_data[1][image_num]

transform = T.ToPILImage()
img = transform(tensor_1)
img.show()
img = transform(tensor_2)
img.show()

