#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from SGD import *




class basic_neural_net(torch.nn.Module):
	def __init__(self, num_input, num_hidden):
		super(basic_neural_net, self).__init__()

		self.l1 = torch.nn.Linear(num_input, num_hidden, bias=True)
		self.l2 = torch.nn.Linear(num_hidden, num_hidden, bias=True)
		self.l3 = torch.nn.Linear(num_hidden, 1, bias=True)
		self.bN = torch.nn.BatchNorm1d(num_hidden)
		self.bn = torch.nn.BatchNorm1d(num_input)

		self.criterion = torch.nn.MSELoss(size_average=False)

		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal(param.data , a=0, mode='fan_in')	
			else:
				pass
				#param.data = torch.zeros(param.data.size())


	def get_optimizer(self, learning_rate):
		#return torch.optim.SGD(self.parameters(), lr=learning_rate)
		return torch.optim.Adam(self.parameters(), lr=learning_rate)
		#return SGD(self, lr=learning_rate)
		#return torch.optim.RMSprop(self.parameters(), lr=learning_rate)
		#return torch.optim.Adagrad(self.parameters(), lr=learning_rate, lr_decay=0.01, weight_decay=0.1)
		#return torch.optim.LBFGS(self.parameters(), lr=learning_rate)
		

	def compute_loss(self, labels, y_pred):
		#return (y_pred - labels).norm()
		return self.criterion(y_pred, labels)

	def forward(self, x):
		#x1 = self.bn(x)

		y1 = F.relu(self.l1(x))
		#y2 = self.bN(y1)
		y3 = self.l2(y1)
		#y4 = self.bN(y3)
		y_pred = self.l3(y3)
		
		return y_pred
