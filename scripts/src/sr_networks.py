"""
File which holds network architectures
"""

import torch
from torch import nn

class Srcnn(torch.nn.Module):
	def __init__(self, f_1=9, f_2=1, f_3=5):
		super(Srcnn, self).__init__()

		self.stack = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=f_1, stride=1),
			nn.ReLU(),

			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=f_2, stride=1),
			nn.ReLU(),

			nn.Conv2d(in_channels=32, out_channels=3, kernel_size=f_3, stride=1)
			)
		
	def forward(self, x):
		return self.stack(x)