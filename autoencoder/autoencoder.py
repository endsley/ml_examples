#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np

class autoencoder():
	def __init__(self, X, D_out):
		self.dtype = torch.FloatTensor
		self.data = X
		self.N = X.shape[0]
		self.D_in = X.shape[1]
		self.D_out = D_out
		tmpX = torch.from_numpy(X)
		self.X = Variable(tmpX.type(self.dtype), requires_grad=False)

		layers = int(np.floor(np.log(self.D_in)/np.log(self.D_out)))
		self.W = {}
		
		
		layer_id = 0
		for m in range(layers):
			Din = int(self.D_in/np.power(2.0,m))
			Dout = int(self.D_in/np.power(2.0,m+1))
			self.W[layer_id] = Variable(torch.randn(Din, Dout).type(self.dtype), requires_grad=True)
			layer_id += 1
		
		self.W[layer_id] = Variable(torch.randn(Dout, self.D_out).type(self.dtype), requires_grad=True)
		layer_id += 1
		self.W[layer_id] = Variable(torch.randn(self.D_out, Dout).type(self.dtype), requires_grad=True)
		layer_id += 1
		
		for m in range(layers)[::-1]:
			Din = int(self.D_in/np.power(2.0,m+1))
			Dout = int(self.D_in/np.power(2.0,m))
				
			self.W[layer_id] = Variable(torch.randn(Din, Dout).type(self.dtype), requires_grad=True)
			layer_id += 1
			
	def print_W_shapes(self):
		for i,j in self.W.items():	
			print i , j.data.numpy().shape


	def forward(self):
		h = self.X
		for i, j in self.W.items():
			h = h.mm(self.W[i])

			if i < (len(self.W)-1):
				h = h.clamp(min=0)

		loss = (h - self.X).pow(2).sum()
		return loss

	def step_forward(self, lr):
		loss = self.forward()
		loss.backward()

		for i, j in self.W.items():
			self.W[i].data -= lr * self.W[i].grad.data

		loss_after = self.forward()
		return [loss, loss_after]

	def step_back(self, lr):
		for i, j in self.W.items():
			self.W[i].data += lr * self.W[i].grad.data

	def optimize(self):
		lr = 1e-3

		for m in range(1000):
			[loss, loss_after] = self.step_forward(lr)

			if loss_after.data[0] > loss.data[0]: 
				self.step_back(lr)
				lr = lr * 0.6
			else:
				lr = lr * 1.05

			print loss.data[0], lr




			#loss = self.forward()
			#loss.backward()

			#print loss.data[0]
			#for i, j in self.W.items():
			#	self.W[i].data -= lr * self.W[i].grad.data
			#	self.W[i].grad.data.zero_()


x = np.random.randn(30,20)
#x = np.ones((10,9))
AE = autoencoder(x,4)
AE.optimize()

