"""
File which holds super resolution network architectures
"""

import torch
from torch import nn

class Srcnn(torch.nn.Module):
	def __init__(self, f_1=9, f_2=1, f_3=5):
		super(Srcnn, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=f_1, padding=f_1//2),
			nn.ReLU(),

			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=f_2, padding=f_2//2 ),
			nn.ReLU(),

			nn.Conv2d(in_channels=32, out_channels=3, kernel_size=f_3, padding=f_3//2)
			)
		
	def forward(self, x):
		return self.stack(x)
	
class VDsrcnn(torch.nn.Module):
	def __init__(self, kernel=3, distance=6):
		super(VDsrcnn, self).__init__()

		self.distance = distance

		self.init_conv = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel, padding=kernel//2),
			nn.ReLU(),
			)
		
		self.mid_conv = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, padding=kernel//2),
			nn.ReLU(),
			)
		
		self.end_conv = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel, padding=kernel//2),
			)

	def forward(self, x):
		input_x = x
		x = self.init_conv(x)

		for i in range(self.distance - 2):
			x = self.mid_conv(x)

		x = self.end_conv(x)
		
		y = x + input_x
		return y
