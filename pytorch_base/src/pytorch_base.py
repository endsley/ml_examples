#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np


class pytorch_base():
	def __init__(self, db):
		self.db = db

	def optimize(self):
		db = self.db
		lr = db['learning_rate']
		optimizer = db['model'].get_optimizer(lr)

		for epoch in range(db['epoc_loop']):
			for i, data in enumerate(db['data_loader'], 0):
				inputs, labels = data

				inputs = Variable(inputs.type(db['dataType']), requires_grad=False)
				labels = Variable(labels.type(db['dataType']), requires_grad=False)

				y_pred = db['model'](inputs)
				loss = db['model'].compute_loss(labels, y_pred)
	
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	
				if db['print_loss']: print loss.data[0]
				if loss.data[0] < 0.001: break;






