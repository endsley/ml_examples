#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
from torch.autograd import Variable
import numpy as np
import collections
import pickle

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
		self.loss_fn = torch.nn.MSELoss(size_average=False)

		
		[self.encoder, encodeElements] = self.build_encoder(n_encoding_layers)
		[self.decoder, decodeElements] = self.build_decoder(n_encoding_layers)

		self.jointParams = [p for p in self.encoder.parameters()] + [p for p in self.decoder.parameters()]


	def build_decoder(self, n_encoding_layers):
		layer_list = []
		nodeDiff = self.D_in - self.D_out
		layer_decrease = int(nodeDiff/float(n_encoding_layers))

		l = self.D_in
		layer_index = 0
		for i in range(n_encoding_layers):
			l2 = l - layer_decrease

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l2, l, bias=True))
			layer_index += 1
			layer_list.append(linLayer)

			if i < (n_encoding_layers-1):
				Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
				layer_index += 1
				layer_list.append(Relayer)

			l = l2

		if l > self.D_out:
			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(self.D_out, l, bias=True))
			layer_index += 1
			layer_list.append(linLayer)


		layer_list = layer_list[::-1]
		OD = collections.OrderedDict(layer_list)
		decoder = torch.nn.Sequential(OD)
	
		return [decoder, layer_list] 



	def build_encoder(self, n_encoding_layers):
		layer_list = []
		nodeDiff = self.D_in - self.D_out
		layer_decrease = int(nodeDiff/float(n_encoding_layers))

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

		if l > self.D_out:
			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, self.D_out, bias=True))
			layer_index += 1
			layer_list.append(linLayer)


		OD = collections.OrderedDict(layer_list)
		encoder = torch.nn.Sequential(OD)
		
		return [encoder, layer_list]


	def show_autoencoder_layers(self):
		for m in self.layer_list: print m


	def get_optimizer(self, learning_rate, opt='Adam'):
		if opt == 'Adam': return torch.optim.Adam(self.jointParams, lr=learning_rate)
		#if opt == 'Adamax': return torch.optim.Adamax(model.parameters(), lr=learning_rate)
		#if opt == 'ASGD': return torch.optim.ASGD(model.parameters(), lr=learning_rate)
		#if opt == 'SGD': return torch.optim.SGD(model.parameters(), lr=learning_rate)

	def forward_autoencoder(self):
		encode_X = self.encoder(self.X)
		y_pred = self.decoder(encode_X)
		return self.loss_fn(y_pred, self.X)


	def optimize_autoencoder(self):
		lr = 1e-2
		optimizer = self.get_optimizer(lr, opt='Adam')

		loss = self.forward_autoencoder()

		for t in range(500000):
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		

			new_loss = self.forward_autoencoder()
			lossDiff = loss.data[0] - new_loss.data[0]

			if lossDiff <= 0.00000000001:
				lr = lr*0.7
				optimizer = self.get_optimizer(lr, opt='Adam')
	
			if lr < 0.0000000001: break;
			if new_loss.data[0] < 0.01: break;
			loss = new_loss


		return loss
		#encode_X = self.encoder(self.X)
		#print encode_X

	def save_result(self):
		result = { "lion": "yellow", "kitty": "red" }
		pickle.dump( favorite_color, open( "save.p", "wb" ) )


def train():
	x = np.random.randn(30, 5)
	AE = autoencoder(x, 2, 2)
	loss = AE.optimize_autoencoder()
	
	if os.path.exists('results.pk'):
		res = pickle.load(open('results.pk','rb'))
		if loss.data[0] < res['loss']:
			res['loss'] = loss.data[0]
			res['autoencoder'] = AE
			print 'Lowest loss : ' , res['loss']
	else:
		res = {}
		res['loss'] = loss.data[0]
		res['autoencoder'] = AE
	
	pickle.dump(res, open('results.pk','wb'))

def load():
	if os.path.exists('results.pk'):
		res = pickle.load(open('results.pk','rb'))

		print res['loss']
		print res['autoencoder']
		
load()
