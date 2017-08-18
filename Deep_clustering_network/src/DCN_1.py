#!/usr/bin/python

#	Tr(Ku H D^(-1/2) K D^(-1/2) H )
#	K = LL^T : phi(X) = L

import math
import time 
import sklearn.metrics
import torch
from torch.autograd import Variable
#import autograd.numpy as np
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy
import sklearn.metrics
import numpy.matlib
import matplotlib.pyplot as plt


class DCN:
	def __init__(self, data_set, k, run_name, hidden_node_count=10):
		self.X = data_set
		self.k = k
		self.N = data_set.shape[0]
		self.d = data_set.shape[1]
		self.hidden_d = hidden_node_count
		self.output_d = k							
		self.run_name = run_name

		self.U_matrix = np.random.random([self.N,self.k])
		self.H_matrix = np.eye(self.N) - np.ones((self.N,self.N))/self.N
		self.I = np.eye(self.N)

		self.lambdaV = -100
		self.alpha = 0.001
	
		self.dtype = torch.FloatTensor
		self.xTor = torch.from_numpy(data_set)
		self.xTor = Variable(self.xTor.type(self.dtype), requires_grad=False)


		# Random Fourier Features	
		self.sample_num = 100
		u = np.random.rand(self.output_d, self.sample_num)

		b = np.random.rand(1, self.sample_num)
		b = np.matlib.repmat(b, self.N, 1)

		self.phase_shift = torch.from_numpy(b)
		self.phase_shift = Variable(self.phase_shift.type(self.dtype), requires_grad=False)


		np.set_printoptions(precision=3)
		np.set_printoptions(threshold=np.nan)
		np.set_printoptions(linewidth=300)
		np.set_printoptions(suppress=True)


	def initialize_W_to_Gaussian(self):
		noise = 0.05*np.min(np.absolute(self.X))
		
		for m in range(len(self.NN._modules)):
			if m == 0:
				#column = np.zeros((self.hidden_d, 1))
				column = noise*np.random.randn(self.hidden_d, 1)
				column[0,0] = 1
				column[1,0] = -1
		
				total_column = np.copy(column)
		
				for n in range(self.d-1):
					new_column = np.roll(column, 2*(n+1))
					total_column = np.hstack((total_column, new_column))
		
				total_column = total_column.astype(np.float32)
				self.NN[m].weight.data = torch.from_numpy(total_column)
				self.NN[m].bias.data.fill_(0)
			elif(m == (len(self.NN._modules)-1) and (type(self.NN[m]) == torch.nn.Linear)):
				column = np.zeros((self.hidden_d, 1))
				column[0,0] = 1
				column[1,0] = -1
				total_column = np.copy(column)
		
				for n in range(self.output_d-1):
					new_column = np.roll(column, 2*(n+1))
					total_column = np.hstack((total_column, new_column))
		
				total_column = total_column.astype(np.float32).T
				self.NN[m].weight.data = torch.from_numpy(total_column)
				self.NN[m].bias.data.fill_(0)
		
			elif(type(self.NN[m]) == torch.nn.Linear):
				self.NN[m].bias.data.fill_(0)
				self.NN[m].weight.data = torch.eye(self.hidden_d)
		
		
		#for m in range(len(self.NN._modules)):
		#	if(type(self.NN[m]) == torch.nn.Linear):
		#		print self.NN[m].bias.data
		#		print self.NN[m].weight.data
		#		print '\n'






	def draw_heatMap(self, mtrix):
		try:
			plt.imshow(mtrix.data.numpy(), cmap='Blues', interpolation='nearest')
			plt.show()
		except:
			plt.imshow(mtrix, cmap='Blues', interpolation='nearest')
			plt.show()


	def compute_Linear_Laplacian(self, input_data):
		K = torch.mm(input_data,input_data.transpose(0,1))

		D1 = torch.sqrt(1/K.sum(1))
		D = torch.mm(D1, D1.transpose(0,1))	
		L = K*D

		return L

	def compute_Gaussian_Laplacian(self, input_data):
#		# Use random fourier features
#		d = input_data.data.numpy().shape[1]
#		u = np.random.rand(d , self.sample_num)
#		self.rand_proj = torch.from_numpy(u)
#		self.rand_proj = Variable(self.rand_proj.type(self.dtype), requires_grad=False)
#		import pdb; pdb.set_trace()
#
#		P = torch.cos(torch.mm(input_data,self.rand_proj) + self.phase_shift)
#		K = torch.mm(P, P.transpose(0,1))
#		K = (2.0/self.sample_num)*K
#
#		D1 = torch.sqrt(1/K.sum(1))
#		D = torch.mm(D1, D1.transpose(0,1))	
#		L = K*D



		# Use actual gaussian kernel
		K = torch.FloatTensor(self.N, self.N)
		K = Variable(K.type(self.dtype), requires_grad=False)
		Y = input_data
		import pdb; pdb.set_trace()
		#sigma = np.median(sklearn.metrics.pairwise.pairwise_distances(Y.data.numpy()))/2
		#print 'sigma : ' , sigma
		sigma = 2

		for i in range(self.N):
			for j in range(self.N):
				tmpY = (Y[i,:] - Y[j,:]).unsqueeze(0)
				eVal = -(torch.mm(tmpY, tmpY.transpose(0,1)))/(2*sigma*sigma)
				K[i,j] = torch.exp(eVal)

		D1 = torch.sqrt(1/K.sum(1))
		D = torch.mm(D1, D1.transpose(0,1))	
		L = K*D


		return L

	def apply_centering(self, sqMatrix):
		sqMatrix = sqMatrix.data.numpy()
		centered = self.H_matrix.dot(sqMatrix).dot(self.H_matrix)
		return centered

	def calc_U(self, input_kernel):
		eigenValues,eigenVectors = np.linalg.eig(input_kernel)

		idx = eigenValues.argsort()
		idx = idx[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]
	
		previous_U = np.copy(self.U_matrix)
		self.U_matrix = eigenVectors[:,:self.k]

		return self.U_matrix
		#self.change_in_U = np.linalg.norm(previous_U - self.U_matrix)/np.linalg.norm(previous_U)

	def create_Phi(self, U):
		phi = U.dot(U.T)			# kernel U
		phi = torch.from_numpy(phi)
		phi = Variable(phi.type(self.dtype), requires_grad=False)
		return phi

	def forward_pass(self, phi):		#	Gaussian Version
		Y = self.NN(self.xTor)	
	
		#L = self.compute_Linear_Laplacian(Y)
		L = self.compute_Gaussian_Laplacian(Y)

		cost = -(phi*L).sum() 
		return cost

	def update_W(self, U):
		learning_rate = 1
		phi = self.create_Phi(U)

		for lmda_count in range(1):
			while True:
		
				#lmda = torch.from_numpy(self.lmda_hold)
				#lmda = Variable(lmda.type(self.dtype), requires_grad=False)

				cost = self.forward_pass(phi)
				
				self.NN.zero_grad()
				cost.backward()
			
				while True:		#	Adaptive Learning Rate
					for param in self.NN.parameters():
						param.data -= learning_rate * param.grad.data
		
					new_cost = self.forward_pass(phi)
					#print 'new cost : ', new_cost.data.numpy()[0], type(new_cost.data.numpy()[0])


					while str(new_cost.data.numpy()[0]) == 'nan': 	# if reached a bad point re-initialize
						for param in self.NN.parameters():
							try:
								param.data = 0.1*torch.randn(param.data.numpy().shape[0],param.data.numpy().shape[1])
							except:
								param.data = 0.1*torch.randn(param.data.numpy().shape[0])
						new_cost = self.forward_pass(phi)
						print '----------------  Reinitialized '


					if(new_cost.data[0] > cost.data[0]): # if got worse, undo and lower the learning rate. 
						for param in self.NN.parameters():
							param.data += learning_rate * param.grad.data
	
						learning_rate = learning_rate*0.6
					else: 
						learning_rate = learning_rate*1.01
						break
					if learning_rate < 0.00000001: break
	
	
				grad_norm = 0	
				for param in self.NN.parameters():
					grad_norm += param.grad.data.norm()
	
				print(learning_rate, ' , ' , cost.data[0], ' , ' , grad_norm)
				
				if grad_norm < 0.01: print('Gradient Exit'); break
				if (np.absolute(new_cost.data.numpy() - cost.data.numpy()))/np.absolute(new_cost.data.numpy()) < 0.0001: print('Cost Exit'); break;
				if learning_rate < 0.0000001: print('Learning Rate Exit'); break


				#Y = self.NN(self.xTor)
				#new_sigma = np.median(sklearn.metrics.pairwise.pairwise_distances(Y.data.numpy()))
				#print 'sigma : ' , new_sigma
				#L = self.compute_Gaussian_Laplacian(Y)
				#self.draw_heatMap(L)
				#import pdb; pdb.set_trace()

		self.forward_pass(phi)
		Y = self.NN(self.xTor)
		L = self.compute_Gaussian_Laplacian(Y)
		import pdb; pdb.set_trace()
		return L	

	def run(self):
		L = self.compute_Gaussian_Laplacian(self.xTor)

		L = self.apply_centering(L)

		U = self.calc_U(L)

		Ku = U.dot(U.T)
		original_cost = (Ku*L).sum()
		print '\noriginal cost : ' , -original_cost

		L = self.update_W(U)
		L = self.apply_centering(L)
		U = self.calc_U(L)
		U = normalize(U, norm='l2', axis=1)

		allocation = KMeans(self.k).fit_predict(U)

		print allocation
		Ku = U.dot(U.T)
		final_cost = (Ku*L).sum()

		print 'final cost : ' , final_cost 
		#import pdb; pdb.set_trace()
		return allocation


