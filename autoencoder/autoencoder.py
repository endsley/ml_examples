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
		layer_decrease = int(nodeDiff/float(n_encoding_layers))
		#layer_decrease = 0
		layer_list = []

		l = self.D_in
		layer_index = 0
		for i in range(n_encoding_layers):
			l2 = l - layer_decrease

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, l2, bias=True))
			layer_index += 1
			layer_list.append(linLayer)

			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			l = l2

		if l > D_out:
			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, D_out, bias=True))
			layer_index += 1
			layer_list.append(linLayer)
			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(D_out, l, bias=True))
			layer_index += 1
			layer_list.append(linLayer)
			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)


		for i in range(n_encoding_layers):
			l2 = l + layer_decrease

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, l2, bias=True))
			layer_index += 1
			layer_list.append(linLayer)

			if i < (n_encoding_layers-1):
				Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
				layer_index += 1
				layer_list.append(Relayer)

			l = l2

		OD = collections.OrderedDict(layer_list)
		self.NN = torch.nn.Sequential(OD)
		self.layer_list = layer_list


	def show_autoencoder_layers(self):
		for m in self.layer_list: print m


	def get_optimizer(self, learning_rate, opt='Adam'):
		if opt == 'Adam': return torch.optim.Adam(self.NN.parameters(), lr=learning_rate)
		if opt == 'Adamax': return torch.optim.Adamax(model.parameters(), lr=learning_rate)
		if opt == 'ASGD': return torch.optim.ASGD(model.parameters(), lr=learning_rate)
		if opt == 'SGD': return torch.optim.SGD(model.parameters(), lr=learning_rate)

	def optimize(self):
		loss_fn = torch.nn.MSELoss(size_average=False)
		learning_rate = 1e-6
		optimizer = self.get_optimizer(learning_rate, opt='Adam')
		

		y_pred = self.NN(self.X)
		loss = loss_fn(y_pred, self.X)

		for t in range(500000):
			#self.NN.zero_grad()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
			#for param in self.NN.parameters():
			#	param.data -= learning_rate * param.grad.data
		
			print loss.data[0]

			y_pred = self.NN(self.X)
			new_loss = loss_fn(y_pred, self.X)
			lossDiff = np.absolute(loss.data[0] - new_loss.data[0])

			#if(lossDiff < 0.0000000000001): break;
			if new_loss.data[0] < 0.01: break;
			loss = new_loss





#x = np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
#y = np.array([[0.3],[0.3],[0.6],[0.6],[1],[1]])
x = np.random.randn(30, 20)

#x = np.ones((10,9))
AE = autoencoder(x, 4, 2)
AE.optimize()

