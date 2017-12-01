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
		Dloader = db['data_loader']

		for epoch in range(db['epoc_loop']):
			for i, data in enumerate(Dloader, 0):
				inputs, labels = data
				
				inputs = Variable(inputs.type(db['dataType']), requires_grad=False)
				labels = Variable(labels.type(db['dataType']), requires_grad=False)

				y_pred = db['model'](inputs)
				loss = db['model'].compute_loss(labels, y_pred)

				db['model'].zero_grad()	
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


				#lr = lr*0.999
				#optimizer = db['model'].get_optimizer(lr)
				#print lr

				#if loss.data[0] < 0.01: Dloader = self.db['data_loader_full']


			#if np.random.rand() > 0.99:
			#	lr = lr*0.90
			#	print '\t-----------------' , lr
			#	#lr = lr / (epoch + 1)
			#	#print '\t' , lr
			#	#for param_group in optimizer.param_groups: param_group['lr'] = lr


			if db['print_loss']: print epoch, loss.data[0]
			if loss.data[0] < 0.001: break;






#def adjust_learning_rate(optimizer, epoch):
#    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = args.lr * (0.1 ** (epoch // 30))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr



