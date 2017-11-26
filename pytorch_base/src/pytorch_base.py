#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np


class pytorch_base():
	def __init__(self, db):
		self.db = db
		self.dtype = torch.FloatTensor

		#self.optimizer = torch.optim.SGD(db['model'].parameters(), lr=0.01)
		#self.optimizer = torch.optim.Adam(db['model'].parameters(), lr=0.005)

	def optimize(self):
		db = self.db
		lr = db['learning_rate']
		optimizer = db['model'].get_optimizer(lr)

		for epoch in range(db['epoc_loop']):
			for i, data in enumerate(db['data_loader'], 0):
				inputs, labels = data

				inputs = Variable(inputs.type(self.dtype), requires_grad=False)
				labels = Variable(labels.type(self.dtype), requires_grad=False)

				y_pred = db['model'](inputs)
				loss = db['model'].compute_loss(labels, y_pred)
	
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
	
				if db['print_loss']: print loss.data[0]
				if loss.data[0] < 0.001: break;








			#inputs = Variable(db['dataset'].x.type(self.dtype), requires_grad=False)
			#labels = Variable(db['dataset'].y.type(self.dtype), requires_grad=False)

			#y_pred = db['model'](inputs)
			#loss = db['model'].compute_loss(labels, y_pred)
			#db['model'].zero_grad()
			#loss.backward()

  			#for param in db['model'].parameters():	# Gradient descent
  			#	param.data -= 0.001 * param.grad.data



#	def optimize(self):
#		db = self.db
#
#		for epoch in range(db['epoc_loop']):
#			for i, data in enumerate(db['data_loader'], 0):
#				inputs, labels = data
#
#				inputs = Variable(inputs.type(self.dtype), requires_grad=False)
#				labels = Variable(labels.type(self.dtype), requires_grad=False)
#
#				#print inputs, labels
#				#import pdb; pdb.set_trace()
#
#				y_pred = db['model'](inputs)
#				loss = db['model'].compute_loss(labels, y_pred)
#				loss.backward()
#
#
#				for param in db['model'].parameters():	# Gradient descent
#					param.data -= 0.0001 * param.grad.data
#
#
#				print loss.data[0]
#				
##				self.opt.zero_grad()
##				loss.backward()
##				self.opt.step()

