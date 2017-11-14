#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import collections

class autoencoder():
	def __init__(self, X, n_encoding_layers, D_out):	#D_out is the Dimension of bottleneck
		self.dtype = torch.FloatTensor
		self.data = X
		self.N = X.shape[0]
		self.D_in = X.shape[1]
		self.D_out = D_out
		tmpX = torch.from_numpy(X)
		self.X = Variable(tmpX.type(self.dtype), requires_grad=False)
		self.loss_func = torch.nn.MSELoss(size_average=False)

		nodeDiff = self.D_in - D_out
		#layer_decrease = int(nodeDiff/float(n_encoding_layers))
		layer_decrease = 0
		layer_list = []

		l = self.D_in
		layer_index = 0
		for i in range(n_encoding_layers):
			l2 = l - layer_decrease

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, l2, bias=True))
			layer_index += 1
			layer_list.append(linLayer)

			if i < (n_encoding_layers-1):
				Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
				layer_index += 1
				layer_list.append(Relayer)

			l = l2

		print layer_list
		import pdb; pdb.set_trace()
		OD = collections.OrderedDict(layer_list)
		self.NN = torch.nn.Sequential(OD)



	def forward(self):
		h = self.NN(self.X)
		loss = self.loss_func(h, self.X)
		#loss = (h - self.X).pow(2).sum()
		return loss

	def step_forward(self, lr):
		cost = self.forward()
		self.NN.zero_grad()
		cost.backward()

		for param in self.NN.parameters():
			param.data -= lr * param.grad.data
			print 'Norm : ' , np.linalg.norm(param.grad.data.numpy())

		new_cost = self.forward()
		return [cost, new_cost]

	def step_back(self, lr):
		for param in self.NN.parameters():
			param.data += lr * param.grad.data

	def optimize(self):
		loss_fn = torch.nn.MSELoss(size_average=False)
		
		learning_rate = 1e-4
		for t in range(500000):
			y_pred = self.NN(self.X)
			loss = loss_fn(y_pred, self.X)
			print(t, loss.data[0])
			self.NN.zero_grad()
			loss.backward()
		
			for param in self.NN.parameters():
				param.data -= learning_rate * param.grad.data
		
			if loss.data[0] < 0.01:
				print y_pred
				break;
















#		lr = 1
#
#		#self.print_W_shapes()
#
#		while True:
#			[loss, loss_after] = self.step_forward(lr)
#
#			if loss_after.data[0] > loss.data[0]: 
#				self.step_back(lr)
#				lr = lr * 0.6
#			else:
#				lr = lr * 1.15
#
#			print loss.data[0], lr
#
#			if np.absolute(loss_after.data[0] - loss.data[0]) < 0.000000001: break;
#
#
#		import pdb; pdb.set_trace()			












			#loss = self.forward()
			#loss.backward()

			#print loss.data[0]
			#for i, j in self.W.items():
			#	self.W[i].data -= lr * self.W[i].grad.data
			#	self.W[i].grad.data.zero_()

x = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
#y = np.array([[0.3],[0.3],[0.6],[0.6],[1],[1]])
#x = np.random.randn(20,8)

#x = np.ones((10,9))
AE = autoencoder(x,2, 2)
AE.optimize()

