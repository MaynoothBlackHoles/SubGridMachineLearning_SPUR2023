"""
File which holds network architectures for binary classification
"""

import torch
from torch import nn


class Fully_connected_1layer(torch.nn.Module):
	def __init__(self, box_length, neurons=1000):
		super(Fully_connected_1layer, self).__init__()

		self.stack = nn.Sequential(
			nn.Flatten(),
			nn.Dropout1d(0.2),
			nn.Linear(in_features=box_length**3*5, out_features=neurons),
			nn.ReLU(),
			
			nn.Linear(in_features=neurons, out_features=2),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)



class Fully_connected_2layer(torch.nn.Module):
	def __init__(self, box_length, neurons=1000):
		super(Fully_connected_1layer, self).__init__()

		self.stack = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=box_length**3*5, out_features=neurons),
			nn.ReLU(),
			
			nn.Linear(in_features=neurons, out_features=neurons),
			nn.ReLU(),
			
			nn.Linear(in_features=neurons, out_features=2),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)



class conv1(torch.nn.Module):
	def __init__(self, numChannels=5, classes=2):
		super(conv1, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv3d(in_channels=numChannels, out_channels=20, kernel_size=2, stride=1),
			nn.ReLU(),	
			nn.Conv3d(in_channels=20, out_channels=50, kernel_size=2, stride=1),
			nn.ReLU(),	
			nn.Conv3d(in_channels=50, out_channels=70, kernel_size=2, stride=1),
			nn.ReLU(),	

			nn.Flatten(),
			nn.Linear(in_features=70, out_features=50),
			nn.ReLU(),

			nn.Linear(in_features=50, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)



class Kernel1_conv(torch.nn.Module):
	def __init__(self, box_length, numChannels=5, classes=2):
		super(Kernel1_conv, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv3d(in_channels=numChannels, out_channels=20, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=20, out_channels=40, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Flatten(),
			nn.Linear(in_features=40 * int(round((box_length/4)**3)), out_features=100), #2560
			nn.ReLU(),

			nn.Linear(in_features=100, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)

	

class Kernel1_conv_big(torch.nn.Module):
	def __init__(self, box_length, numChannels=5, classes=2):
		super(Kernel1_conv_big, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv3d(in_channels=numChannels, out_channels=20, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=20, out_channels=40, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=40, out_channels=60, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=60, out_channels=80, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=80, out_channels=100, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(in_channels=100, out_channels=120, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Flatten(),
			nn.Linear(in_features=120, out_features=100),
			nn.ReLU(),

			nn.Linear(in_features=100, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)	
		
    
    
class Kernel1_conv_big1(torch.nn.Module):
	def __init__(self, box_length, numChannels=5, classes=2):
		super(Kernel1_conv_big1, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv3d(in_channels=numChannels, out_channels=20, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=4, stride=4),

			nn.Conv3d(in_channels=20, out_channels=50, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=4, stride=4),

			nn.Conv3d(in_channels=50, out_channels=100, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=4, stride=4),

			nn.Flatten(),
			nn.Linear(in_features=100, out_features=500),
			nn.ReLU(),

			nn.Linear(in_features=500, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)	
		
    
    
class S1(torch.nn.Module):
	def __init__(self, N_0, box_length=64, numChannels=5, classes=2, p_dropout=0.2):
		super(S1, self).__init__()

		self.stack = nn.Sequential(
			#nn.Dropout3d(p_dropout),
			nn.Conv3d(in_channels=numChannels, out_channels=N_0, kernel_size=7, stride=2),
			#nn.BatchNorm3d()
			nn.ReLU(),	

			#nn.Dropout3d(p_dropout),
			nn.Conv3d(in_channels=N_0, out_channels=3*N_0, kernel_size=3, stride=2),
			#nn.BatchNorm3d()
			nn.ReLU(),	

			#KB1

			nn.MaxPool3d(stride=2, kernel_size=2),

			#KB2

			nn.AvgPool3d(kernel_size=2),

			nn.Flatten(),
			nn.Linear(in_features=70, out_features=50),
			nn.ReLU(),

			nn.Linear(in_features=50, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)



class S1_simple(torch.nn.Module):
	def __init__(self, N_0, box_length, numChannels=5, classes=2, p_dropout=0.2):
		super(S1_simple, self).__init__()

		self.stack = nn.Sequential(
			nn.Dropout3d(p_dropout),
			nn.Conv3d(in_channels=numChannels, out_channels=N_0, kernel_size=5, stride=2),
			#nn.BatchNorm3d()
			nn.ReLU(),	
			#nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Dropout3d(p_dropout),
			nn.Conv3d(in_channels=N_0, out_channels=3*N_0, kernel_size=3, stride=2),
			#nn.BatchNorm3d()
			nn.ReLU(),	
			#nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Flatten(),
			nn.Linear(in_features=120, out_features=250),
			nn.ReLU(),

			nn.Linear(in_features=250, out_features=classes),
			nn.Softmax(dim=1)
			)
		
	def forward(self, x):
		return self.stack(x)



class CompactObjectConv(torch.nn.Module):
    """
    This network is designed to learn a subgrid model for star and black hole
    formation. It can be applied to inputs whose hape is (6, Nx, Ny, Nz) where
    Nx, Ny and Nz can be varied. This allows the model to be trained on smaller
    regions and then be applied to larger regions.
    """
    
    def __init__(self):
        super().__init__()
        
        N = 21
        
        self.stack = nn.Sequential(
			nn.Conv3d(in_channels  = 3,
                      out_channels = N,
                      kernel_size  = 1,
                      stride       = 1),
            nn.LeakyReLU(),
            
            nn.Conv3d(in_channels  = N, 
                      out_channels = N, 
                      kernel_size  = 1, 
                      stride       = 1),
			nn.LeakyReLU(),
            
            nn.Conv3d(in_channels  = N, 
                      out_channels = N, 
                      kernel_size  = 1, 
                      stride       = 1),
			nn.LeakyReLU(),
            
 			nn.AdaptiveMaxPool3d((1, 1, 1)),

 			nn.Flatten(),
            
            nn.Linear(in_features = N, out_features = N),
            nn.LeakyReLU(),
            
            nn.Linear(in_features = N, out_features = 2),
            nn.LeakyReLU(),

 			nn.Softmax(dim=1)
			)
    
    def forward(self, x):
        return self.stack(x)