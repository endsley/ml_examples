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

		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
		y1 = F.relu(self.l1(x))
		y_pred = self.l2(y1)
		
		return y_pred
