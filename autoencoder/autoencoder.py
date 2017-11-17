#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
from torch.autograd import Variable
import numpy as np
import collections
import pickle
import sklearn
from sklearn.cluster import SpectralClustering
from numpy import genfromtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import preprocessing



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

			print loss.data[0]
			if lossDiff <= 0.00000000001:
				lr = lr*0.7
				optimizer = self.get_optimizer(lr, opt='Adam')
	
			if lr < 0.0000000001: break;
			if new_loss.data[0] < 0.01: break;
			loss = new_loss


		return loss

def train():
	dataFile = 'breast-cancer'
	num_of_layers = 2
	num_of_output = 8
	FN = 'networks/AutoNN_' + dataFile + '_' + str(num_of_layers) + '_' + str(num_of_output) + '.pk'

	x = genfromtxt('data/' + dataFile + '.csv', delimiter=',')
	x = preprocessing.scale(x)

	AE = autoencoder(x, num_of_layers, num_of_output)
	loss = AE.optimize_autoencoder()
	

	if os.path.exists(FN):
		res = pickle.load(open(FN,'rb'))
		if loss.data[0] < res['loss']:
			res['loss'] = loss.data[0]
			res['autoencoder'] = AE
			print 'Lowest loss : ' , res['loss']
	else:
		res = {}
		res['loss'] = loss.data[0]
		res['autoencoder'] = AE
	
	pickle.dump(res, open(FN,'wb'))

def load():
	dataFile = 'breast-cancer'
	num_of_layers = 2
	num_of_output = 6
	FN = 'networks/AutoNN_' + dataFile + '_' + num_of_layers + '_' + num_of_output + '.pk'

	data = genfromtxt('data/breast-cancer.csv', delimiter=',')
	label = genfromtxt('data/breast-cancer-labels.csv', delimiter=',')
	res = pickle.load(open(FN,'rb'))
	AE = res['autoencoder']
	encodedX = AE.encoder(AE.X)

	d_matrix = sklearn.metrics.pairwise.pairwise_distances(encodedX.data.numpy(), Y=None, metric='euclidean')
	s = np.median(d_matrix)
	Vgamma = 1/(2*s*s)
	spAlloc = SpectralClustering(2, gamma=Vgamma).fit_predict(encodedX.data.numpy())
	nmi_sp = np.around(normalized_mutual_info_score(label, spAlloc), 3)


	kmAlloc = KMeans(2).fit_predict(encodedX.data.numpy())
	nmi_km = np.around(normalized_mutual_info_score(label, kmAlloc), 3)

	print encodedX

	print nmi_sp
	print nmi_km

	print res['loss']
	print res['autoencoder']
	import pdb; pdb.set_trace()
		
train()
#load()
