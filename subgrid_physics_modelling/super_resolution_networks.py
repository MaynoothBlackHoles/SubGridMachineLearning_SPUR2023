"""
File which holds super resolution network architectures
"""

import torch
from torch import nn

class SRcnn(torch.nn.Module):
	def __init__(self, f_1=9, f_2=1, f_3=5):
		super().__init__()

		self.stack = nn.Sequential(
			nn.Conv2d(in_channels  = 3, 
	     			  out_channels = 64,
					  kernel_size  = f_1,
					  padding      = f_1//2
					  ),
			nn.ReLU(),

			nn.Conv2d(in_channels  = 64,
	     			  out_channels = 32,
					  kernel_size  = f_2,
					  padding      = f_2//2
					  ),
			nn.ReLU(),

			nn.Conv2d(in_channels  = 32, 
	     			  out_channels = 3,
					  kernel_size  = f_3,
					  padding      = f_3//2
					  )
			)
		
	def forward(self, x):
		return self.stack(x)
	
class VDSRcnn(torch.nn.Module):
	def __init__(self, kernel=3, depth=6):
		super().__init__()

		self.depth = depth

		self.init_conv = nn.Sequential(
			nn.Conv2d(in_channels  = 3,
	     			  out_channels = 64,
					  kernel_size  = kernel,
					  padding      = kernel//2
					  ),
			nn.ReLU(),
			)
		
		self.mid_conv = nn.Sequential(
			nn.Conv2d(in_channels  = 64,
	     			  out_channels = 64,
					  kernel_size  = kernel,
					  padding      = kernel//2
					  ),
			nn.ReLU(),
			)
		
		self.end_conv = nn.Sequential(
			nn.Conv2d(in_channels=64,
	     			  out_channels=3,
					  kernel_size=kernel,
					  padding=kernel//2
					  ),
			)

	def forward(self, x):
		input_x = x
		x = self.init_conv(x)

		for i in range(self.depth - 2):
			x = self.mid_conv(x)

		x = self.end_conv(x)
		
		y = x + input_x
		return y

class Residual_CNN_3D(torch.nn.Module):
	def __init__(self, depth=3, channels=6, mid_channels=64, kernel_front=3, kernel_mid=3, kernel_end=3):
		super().__init__()

		self.depth = depth

		self.init_conv = nn.Sequential(
			nn.Conv3d(in_channels  = channels,
					  out_channels = mid_channels,
					  kernel_size  = kernel_front,
					  padding      = kernel_front//2
					  ),
			nn.ReLU()
			)
		
		self.mid_conv = nn.Sequential(
			nn.Conv3d(in_channels  = mid_channels,
	     			  out_channels = mid_channels,
					  kernel_size  = kernel_mid,
					  padding      = kernel_mid//2
					  ),
			nn.ReLU()
			)
		
		self.end_conv = nn.Sequential(
			nn.Conv3d(in_channels  = mid_channels, 
	     			  out_channels = channels,
					  kernel_size  = kernel_end,
					  padding      = kernel_end//2
					  )
			)

	def forward(self, x):
		x = self.init_conv(x)

		for i in range(self.depth - 2):
			x = self.mid_conv(x)

		x = self.end_conv(x)
		
		y = x + x
		return y
	
class CNN_3D(torch.nn.Module):
	def __init__(self, depth=3, channels=1, mid_channels=16, kernel_front=3, kernel_mid=3, kernel_end=3):
		super().__init__()

		self.depth = depth

		self.init_conv = nn.Sequential(
			nn.Conv3d(in_channels  = channels,
					  out_channels = mid_channels,
					  kernel_size  = kernel_front,
					  padding      = kernel_front//2
					  ),
			nn.ReLU()
			)
		
		self.mid_conv = nn.Sequential(
			nn.Conv3d(in_channels  = mid_channels,
	     			  out_channels = mid_channels,
					  kernel_size  = kernel_mid,
					  padding      = kernel_mid//2
					  ),
			nn.ReLU()
			)
		
		self.end_conv = nn.Sequential(
			nn.Conv3d(in_channels  = mid_channels, 
	     			  out_channels = channels,
					  kernel_size  = kernel_end,
					  padding      = kernel_end//2
					  )
			)

	def forward(self, x):
		x = self.init_conv(x)

		for i in range(self.depth - 2):
			x = self.mid_conv(x)

		x = self.end_conv(x)
		
		return x