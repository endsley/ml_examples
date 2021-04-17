#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import math

import numpy as np
import sys
import os
import sklearn.metrics
from numpy import genfromtxt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

torch.set_printoptions(edgeitems=3)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=2)
torch.set_printoptions(linewidth=400)



def load_data(data_name, prefix='data/'):
	X = np.loadtxt(prefix + data_name + '.csv', delimiter=',', dtype=np.float64)
	X = preprocessing.scale(X)
	Y = np.loadtxt(prefix + data_name + '_label.csv', delimiter=',', dtype=np.int32)			

	if os.path.exists(prefix + data_name + '_test.csv'):
		Xⲧ = np.loadtxt(prefix + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
		Xⲧ = preprocessing.scale(Xⲧ)
	else: Xⲧ = None

	if os.path.exists(prefix + data_name + '_label_test.csv'):
		Yⲧ = np.loadtxt(prefix + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			
	else: Yⲧ = None

	return [X,Y,Xⲧ,Yⲧ]

#	Take data and out equal number of samples from each class a batch
class balance_class_sampling:
	def __init__(self, X, Y, with_replacement=True):
		self.X = X
		self.Y = Y
		self.d = X.shape[1]
		self.n = X.shape[0]
		self.epoch_count = 0

		if type(Y) == type([]): self.Y = np.array(Y)
		self.Y = np.reshape(self.Y,(len(Y),1))
		self.Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(self.Y)
		self.c = self.Yₒ.shape[1]

		self.X_list = {}
		self.X_shuffled = {}
		self.Y_list = {}
		self.Yₒ_list = {}

		self.l = np.unique(Y)
		for i in self.l:
			indices = np.where(Y == i)[0]
			self.X_list[i] = X[indices, :]
			self.Y_list[i] = self.Y[indices]
			self.Yₒ_list[i] = self.Yₒ[indices, :]

			self.X_shuffled[i] = shuffle(self.X_list[i])		

		#for each class, gen a new X[]
		#each time we sample, we pick a subset from each class 
		#combine the subset into 1 X and 1 Y, output that subset

	def epoch_callback(self, epoch):		# this function needs to be overwritten by an "outside" function.
		pass

	def sample(self, samples_per_class=10):
		if samples_per_class > self.n:
			print('Error : Your batch size is more than the number of samples\n')
			sys.exit()

		self.epoch_increased = False
		# Check if each class has enough samples per class
		for i in self.l:
			if self.X_shuffled[i].shape[0] < samples_per_class:
				self.epoch_increased = True
				self.epoch_count += 1
				self.epoch_callback(self.epoch_count)
				for j in self.l:		# reshuffle each class
					self.X_shuffled[j] = shuffle(self.X_list[j])		
				break
			


		Xout = np.empty((0, self.d))	
		Yout = np.empty((0,1))	
		Yₒout = np.empty((0, self.c))	

		for i in self.l:
			newX = self.X_shuffled[i][0:samples_per_class, :]
			self.X_shuffled[i] = self.X_shuffled[i][samples_per_class:, :]

			Xout = np.vstack((Xout, newX))
			Yout = np.vstack((Yout, self.Y_list[i][0:samples_per_class]))
			Yₒout = np.vstack((Yₒout, self.Yₒ_list[i][0:samples_per_class, :]))

		return Xout, Yout, Yₒout



class hsic_net(torch.nn.Module):
	def __init__(self, X):
		"""
		Find the optimal σ for Gaussian
		"""
		super().__init__()

		if torch.cuda.is_available(): self.device = 'cuda'
		else: self.device = 'cpu'

		Ðᵧ = sklearn.metrics.pairwise.pairwise_distances(X)
		σ = np.median(Ðᵧ)
		self.σ = torch.nn.Parameter(torch.tensor([σ], device=self.device))	
		
		self.HKᵧ = None

	def forward(self, X, Yₒ):
		γ = 1.0 / (2 * self.σ * self.σ)
		K = torch.exp(γ * -torch.norm(X.unsqueeze(1) - X, dim=2) ** 2)
		#print(K)
	
		Kᵧ = Yₒ@Yₒ.T
		self.HKᵧ = HKᵧ = Kᵧ - torch.mean(Kᵧ, axis=0)

		#if self.HKᵧ is None:
		#	Kᵧ = Yₒ@Yₒ.T
		#	self.HKᵧ = HKᵧ = torch.tensor(Kᵧ - np.mean(Kᵧ, axis=0))
		#else:
		#	HKᵧ = self.HKᵧ

		HKᵪ = K - torch.mean(K, dim=0) 	
		Hᵪᵧ = torch.sum(HKᵪ.T*HKᵧ)
		return -Hᵪᵧ				# we want to maximize this, so set it to negative


def stochastic_opt_σ(X,Y, samples_per_class, max_num_epochs=3000, max_loop=40000, lr=1e-1, verbose=False, print_every_other_n_epochs=20):
	if torch.cuda.is_available(): device = 'cuda'
	else: device = 'cpu'
	HN = hsic_net(X).to(device)

	BCS = balance_class_sampling(X,Y)
	num_of_classes = len(np.unique(Y))

	optimizer = torch.optim.Adam(HN.parameters(), lr=lr)	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=verbose)

	hsic_list = []
	for i in range(max_loop):
		[Xout, Yout, Yₒout] = BCS.sample(samples_per_class=samples_per_class)
		if BCS.epoch_count > max_num_epochs: break

		Xout = torch.tensor(Xout).to(device)
		Yₒout = torch.tensor(Yₒout).to(device)

		#print(Xout, '\n--------------\n', Yₒout, '\n\n')
		#import pdb; pdb.set_trace()

		hsic = HN(Xout, Yₒout)
		hsic_list.append(-hsic.item())
		if BCS.epoch_increased and BCS.epoch_count%print_every_other_n_epochs == 0: 
			avg_epoch_hsic = np.mean(hsic_list)
			scheduler.step(avg_epoch_hsic)
			print('Epoch : %d, Avg HSIC: %.3f, σ: %.3f'%(BCS.epoch_count, avg_epoch_hsic, HN.σ))
		
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		hsic.backward()
		optimizer.step()

	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.atleast_2d(Y).T)
	Yₒ = torch.tensor(Yₒ).to(device)
	X = torch.tensor(X).to(device)

	total_hsic = -HN(X, Yₒ)
	return [HN.σ, total_hsic]
		

if __name__ == '__main__':
	#data_name = 'wine'
	data_name = 'spiral_arm'
	[X,Y,Xⲧ,Yⲧ] = load_data(data_name, prefix='../dataset/')

	[best_σ, total_hsic] = stochastic_opt_σ(X,Y, samples_per_class=30, verbose=True)
	print('best_σ: %.3f, final hsic: %.3f'%(best_σ, total_hsic))
	import pdb; pdb.set_trace()

	

