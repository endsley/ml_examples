#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




class basic_neural_net(torch.nn.Module):
	def __init__(self):
		super(basic_neural_net, self).__init__()

		self.l1 = torch.nn.Linear(2, 2, bias=True)
		self.l2 = torch.nn.Linear(2, 1, bias=True)

		self.criterion = torch.nn.MSELoss(size_average=False)

		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal(param.data , a=0, mode='fan_in')	
			else:
				param.data = torch.zeros(param.data.size())


	def get_optimizer(self, learning_rate):
		return torch.optim.Adam(self.parameters(), lr=learning_rate)

	def compute_loss(self, labels, y_pred):
		#return (y_pred - labels).norm()
		return self.criterion(y_pred, labels)

	def forward(self, x):
		y1 = F.relu(self.l1(x))
		y_pred = self.l2(y1)
		
		return y_pred
