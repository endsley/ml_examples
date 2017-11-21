#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np


class pytorch_base():
	def __init__(self, db):
		self.db = db

	def optimize(self):
		db = self.db

#		criterion = torch.nn.BCELoss(size_average=True)
#		optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
		
		for epoch in range(db['epoc_loop']):
		    for i, data in enumerate(db['data_loader'], 0):
		        inputs = Variable(data)
				y_pred = self.model(inputs)
				loss = self.model.compute_loss(y_pred)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

