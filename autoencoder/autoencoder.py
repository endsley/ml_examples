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
import time 

dataFile = 'breast-cancer'
labelFile = 'breast-cancer-labels'
num_of_layers = 1
num_of_output = 2


#dataFile = 'mnist_10000_784'
#labelFile = 'mnist_10000_784_label'
#num_of_layers = 10
#num_of_output = 10


FN = 'networks/AutoNN_' + dataFile + '_' + str(num_of_layers) + '_' + str(num_of_output) + '.pk'


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

		#self.plot_initial_weights()
	

	def plot_initial_weights(self):
		#This part plots out the initialization weights
		M = None
		for params in self.jointParams:
			if len(params.data.size()) == 2:
				if type(M) == type(None):
					M = params.data.numpy()[0]
				else:
					M = np.append(M, params.data.numpy()[0])
	
		import matplotlib.pyplot as plt
		plt.hist(M, normed=True, bins=100)
		plt.ylabel('Probability');
		#plt.title('kaiming_normal');
		plt.show()


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

			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			btn = torch.nn.BatchNorm1d(l2, eps=1e-05, momentum=0.1, affine=False)
			btNorm = ('BatchNorm:' + str(layer_index), btn)
			layer_index += 1
			layer_list.append(btNorm)


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
	

		self.initialize_weights(decoder)

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
				btn = torch.nn.BatchNorm1d(l2, eps=1e-05, momentum=0.1, affine=False)
				btNorm = ('BatchNorm:' + str(layer_index), btn)
				layer_index += 1
				layer_list.append(btNorm)

				#DP = ('Dropout:' + str(layer_index), torch.nn.Dropout(0.1))
				#layer_index += 1
				#layer_list.append(DP)


				Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
				layer_index += 1
				layer_list.append(Relayer)

			l = l2

		if l > self.D_out:
			btn = torch.nn.BatchNorm1d(l2, eps=1e-05, momentum=0.1, affine=False)
			btNorm = ('BatchNorm:' + str(layer_index), btn)
			layer_index += 1
			layer_list.append(btNorm)

			#DP = ('Dropout:' + str(layer_index), torch.nn.Dropout(0.5))
			#layer_index += 1
			#layer_list.append(DP)

			Relayer = ('Relu:' + str(layer_index), torch.nn.ReLU())
			layer_index += 1
			layer_list.append(Relayer)

			linLayer = ('Linear:' + str(layer_index), torch.nn.Linear(l, self.D_out, bias=True))
			layer_index += 1
			layer_list.append(linLayer)


		OD = collections.OrderedDict(layer_list)
		encoder = torch.nn.Sequential(OD)


		self.initialize_weights(encoder)

#		for param in encoder.parameters():
#			if(len(param.data.numpy().shape)) > 1:
#				torch.nn.init.kaiming_normal(param.data , a=0, mode='fan_in')	
#				#torch.nn.init.kaiming_uniform(param.data , a=0, mode='fan_in')	
#				#torch.nn.init.xavier_normal(param.data)
#				#torch.nn.init.xavier_uniform(param.data)
#			else:
#				param.data = torch.zeros(param.data.size())

		return [encoder, layer_list]

	def initialize_weights(self, network):
		for param in network.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal(param.data , a=0, mode='fan_in')	
				#torch.nn.init.kaiming_uniform(param.data , a=0, mode='fan_in')	
				#torch.nn.init.xavier_normal(param.data)
				#torch.nn.init.xavier_uniform(param.data)
			else:
				param.data = torch.zeros(param.data.size())

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

	x = genfromtxt('../dataset/' + dataFile + '.csv', delimiter=',')
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
	data = genfromtxt('../dataset/' + dataFile + '.csv', delimiter=',')
	label = genfromtxt('../dataset/' + labelFile + '.csv', delimiter=',')
	res = pickle.load(open(FN,'rb'))
	AE = res['autoencoder']
	encodedX = AE.encoder(AE.X)

	X = encodedX.data.numpy()
	#X = preprocessing.scale(encodedX.data.numpy())


	d_matrix = sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean')
	s = np.median(d_matrix)
	Vgamma = 1/(2*s*s)
	spAlloc = SpectralClustering(2, gamma=Vgamma).fit_predict(X)
	nmi_sp = np.around(normalized_mutual_info_score(label, spAlloc), 3)


	kmAlloc = KMeans(2).fit_predict(X)
	nmi_km = np.around(normalized_mutual_info_score(label, kmAlloc), 3)

	print X

	print nmi_sp
	print nmi_km

	print res['loss']
	#print res['autoencoder']

	txt = dataFile + ' nmiSP : ' + str(nmi_sp) + ' , nmiKM : ' + str(nmi_km) + ' , num_of_layers:' + str(num_of_layers) + ' , num_of_output:' +  str(num_of_output) + '\n'

	fin = open('auto_out.txt','a')
	fin.write(txt)
	fin.close()
		
	
#for m in range(100):
#	train()

start_time = time.time() 
train()
load()
print("--- %s seconds ---" % (time.time() - start_time))
