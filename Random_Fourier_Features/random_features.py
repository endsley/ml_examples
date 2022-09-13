#!/usr/bin/env python

import numpy as np
import sklearn.metrics
import torch
import numpy.matlib
from torch.autograd import Variable
from numpy.random import rand
import torch.nn.functional as F
from tools import *


#	Written by Chieh Wu
#	This function calculates the Gaussian Kernel by approximate it through Random fourier Feature technique.

class random_feature:
	# sample_num, the larger the better approximation
	# random_feature_method can be 'orthogonal' (default) or 'rff'
	def __init__(self, kernel='rbk', sigma=1, random_feature_method='rff', sample_num=500):
		self.method = random_feature_method
		self.kernel = kernel
		self.m = sample_num		# the number of samples used to approximate k(x,y)
		self.θ = None			# phase shift
		self.σ = sigma

	def initialize_random_features(self, X):
		if self.θ == None:
			self.X = X
			N = X.shape[0]
			d = X.shape[1]
			σˉᑊ = 1/self.σ
			ƻπ = 2*np.pi
			m = self.m

			self.θ = ƻπ*rand(1, m)
	
			if self.method == 'orthogonal':	# perform SORF
				pass
			elif self.method == 'rff':
				self.W = (σˉᑊ)*np.random.randn(d, m)	# random projection matrix W



	def get_feature_map(self, X):
		self.initialize_random_features(X)

		m = self.m
		θ = self.θ
		XW = X.dot(self.W)

		c = np.sqrt(2.0/m)	
		return c*np.cos(XW + θ)	


	def get_kernel(self, X):
		Φ = self.get_feature_map(X)
		K = Φ.dot(Φ.T)
		K = np.maximum(0,K)	# run a relu on the kernel so no negative values
		if self.kernel == 'rbk': K = np.minimum(1,K) # make sure the kernel values doesn't go beyond 1 for gaussian
		return K


#	def initialize_RFF(self, x, sigma, output_torch=False, dtype=None):
#		self.x = x
#		self.N = x.shape[0]
#		self.d = x.shape[1]
#		self.sigma = sigma
#		ƻπ = 2*np.pi
#		
#		if self.phase_shift is not None:
#			if x.shape[0] == self.N: return
#
#		if type(x) == torch.Tensor or type(x) == np.ndarray:	
#			#b = ƻπ*rand(1, self.sample_num)
#			#self.phase_shift = np.matlib.repmat(b, self.N, 1)	
#
#			self.phase_shift = ƻπ*rand(1, self.sample_num)
#
#
#			self.rand_proj = np.random.randn(self.d, self.sample_num)/(self.sigma)
#		else:
#			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(x)))
#
#		if output_torch:
#			self.use_torch(dtype)
#		
#
#
#	def use_torch(self, dtype):
#		self.phase_shift = torch.from_numpy(self.phase_shift)
#		self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)
#
#		self.rand_proj = torch.from_numpy(self.rand_proj)
#		self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)
#
#		diagonal_mask = np.eye(self.N)
#		diagonal_mask = -1*(diagonal_mask - 1)
#		diagonal_mask = torch.from_numpy(diagonal_mask)
#		self.diagonal_mask = Variable(diagonal_mask.type(dtype), requires_grad=False)
#
#
#	def torch_rbf(self):
#		self.xTor = self.x
#		if type(self.x) == np.ndarray:
#			self.xTor = torch.from_numpy(self.x)
#			self.xTor = Variable(self.xTor.type(self.dtype), requires_grad=False)
#
#		if type(self.xTor) != torch.Tensor:
#			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(self.x)))
#
#
#		P = torch.cos(torch.mm(self.xTor,self.rand_proj) + self.phase_shift)
#		K = torch.mm(P, P.transpose(0,1))
#		K = (2.0/self.sample_num)*K
#		K = F.relu(K)
#		K = K*self.diagonal_mask	
#		return K
#
#	def np_feature_map(self, x):
#		ƻπ = 2*np.pi
#		const = np.sqrt(2.0/self.sample_num)	
#		X_after_projection = x.dot(self.rand_proj)
#
#		#X_after_projection + self.phase_shift
#		feature_map = const*np.cos(X_after_projection + self.phase_shift)	
#
#		#nv = np.linalg.norm(feature_map, axis=1)
#		#nv = np.reshape(nv, (len(nv), 1))
#		#feature_map = feature_map/nv
#		
#		return feature_map
#
#	def np_rbf(self):
#		P = np.cos(self.x.dot(self.rand_proj) + self.phase_shift)
#		K = (2.0/self.sample_num)*(P.dot(P.T))
#		K = np.maximum(K, 0)
#		K = np.minimum(K, 1)
#		return K
#
#	def get_rbf(self, x, sigma, output_torch=False, dtype=torch.FloatTensor):
#		self.dtype = dtype
#		self.initialize_RFF(x,sigma, output_torch, dtype)
#
#		if output_torch: return self.torch_rbf()
#		else: return self.np_rbf()


if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)

	X = np.random.randn(5,2)
	σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
	γ = 1.0/(2*σ*σ)

	rff = random_feature(sigma=σ)
	K1 = rff.get_kernel(X)
	K2 = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)

	print_two_matrices_side_by_side(K1, K2, title1='Approx', title2='Real', auto_print=True)


#	rff2 = random_feature(30000)
#	rbf_torch = rff2.get_rbf(X, sigma, True)


#	print(rbf_torch)
#	print('\n')
#	print(rbf_np)
#	print('\n')
#	print(sk_rbf)

