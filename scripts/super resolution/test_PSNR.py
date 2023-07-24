"""
Script to test avarage PSNR of a chosen dataset
"""

import torch

IMAGE_SIZE = 500


def eval_PSNR(x, y):
    MSE = torch.mean(torch.square(x - y))
    MAX_I = torch.max(x)
    PSNR = 20*torch.log10(MAX_I) - 10*torch.log10(MSE)
    return float(PSNR)

def test_PSNR(dataset):
    total_PSNR = 0

    X = dataset[0]        
    Y = dataset[1]        

    size = len(X)

    for i in range(size):
        percentage = round(100 * ((i + 1)/size), 1)
        print(f" Looping {percentage}% done", end="\r")
        total_PSNR += eval_PSNR(X[i], Y[i])

    avg_PSNR = total_PSNR / size
    return avg_PSNR

# load your dataset
train_data = torch.load(f"C:/Users/drozd/Documents/programming stuff/Python Programms/SPUR/super resolution/data/smalltraining_{IMAGE_SIZE}s.pt")

# testing on loaded dataset
data_PSNR = test_PSNR(train_data)

print("-----------------------------")
print(data_PSNR) #32.97324611813414