#!/usr/bin/env python

import numpy as np
import sklearn.metrics
import torch
from torch.autograd import Variable
import torch.nn.functional as F


#	Written by Chieh Wu
#	This function calculates the Gaussian Kernel by approximate it through Random fourier Feature technique.

class RFF:
	# sample_num, the larger the better approximation
	def __init__(self, sample_num=20000):
		self.sample_num = sample_num
		self.phase_shift = None

	def manual_init(self, N, d, sigma, dtype):
		self.phase_shift = 2*np.pi*torch.rand(1, self.sample_num)
		self.phase_shift.repeat(N, 1)
		self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)
		self.rand_proj = torch.randn(d, self.sample_num)/(sigma)
		self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)
		self.N = N

	def initialize_RFF(self, x, sigma, output_torch, dtype):
		if self.phase_shift is not None:
			if x.shape[0] == self.N: return

		if type(x) == torch.autograd.variable.Variable:
			self.x = x
			self.N = x.shape[0]
			self.d = x.shape[1]
			self.sigma = sigma

			self.phase_shift = 2*np.pi*torch.rand(1, self.sample_num)
			self.phase_shift.repeat(self.N, 1)
			self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)
			self.rand_proj = torch.randn(self.d, self.sample_num)/(self.sigma)
			self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)
			return
		

		if type(x) == np.ndarray:
			self.x = x
			self.N = x.shape[0]
			self.d = x.shape[1]
			self.sigma = sigma
	
			b = 2*np.pi*np.random.rand(1, self.sample_num)
			self.phase_shift = np.matlib.repmat(b, self.N, 1)	
			self.rand_proj = np.random.randn(self.d, self.sample_num)/(self.sigma)
		else:
			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(x)))

		if output_torch:
			self.use_torch(dtype)

	def use_torch(self, dtype):
		self.phase_shift = torch.from_numpy(self.phase_shift)
		self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)

		self.rand_proj = torch.from_numpy(self.rand_proj)
		self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)

	def torch_rbf(self, x):
		self.xTor = x
		if type(self.xTor) == np.ndarray:
			self.xTor = torch.from_numpy(self.xTor)
			self.xTor = Variable(self.xTor.type(self.dtype), requires_grad=False)

		if type(self.xTor) != torch.autograd.variable.Variable:
			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(self.xTor)))


		P = torch.cos(torch.mm(self.xTor,self.rand_proj) + self.phase_shift)
		K = torch.mm(P, P.transpose(0,1))
		K = (2.0/self.sample_num)*K
		K = F.relu(K)

		return K

	def numpy_rbf(self):
		P = np.cos(self.x.dot(self.rand_proj) + self.phase_shift)
		K = (2.0/self.sample_num)*(P.dot(P.T))
		K = np.maximum(K, 0)
		K = np.minimum(K, 1)
		return K

	def get_rbf(self, x, sigma, output_torch=False, dtype=torch.FloatTensor):
		self.dtype = dtype
		self.initialize_RFF(x,sigma, output_torch, dtype)

		if output_torch: return self.torch_rbf(x)
		else: return self.numpy_rbf()


if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)

	X = np.random.randn(5,2)

	rff = RFF(30000)
	rbf_np = rff.get_rbf(X, 1)

	rff2 = RFF(30000)
	rbf_torch = rff2.get_rbf(X, 1, True)

	sk_rbf = sklearn.metrics.pairwise.rbf_kernel(X, gamma=0.5)

	print rbf_torch
	print '\n'
	print rbf_np
	print '\n'
	print sk_rbf

