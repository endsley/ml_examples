#!/usr/bin/python

#	Tr(Ku H D^(-1/2) K D^(-1/2) H )
#	K = LL^T : phi(X) = L

import math
import time 
import torch										# version : 0.2.0
from torch.autograd import Variable
#import autograd.numpy as np
import numpy as np									# version : 1.10.4
from sklearn.preprocessing import normalize			# version : 0.17
from sklearn.cluster import KMeans
import sklearn.metrics
import matplotlib 
import matplotlib.pyplot as plt						# version : 1.3.1
import copy
import numpy.matlib

colors = matplotlib.colors.cnames

class DCN:
	def __init__(self, data_set, k, run_name, hidden_node_count=10, sigma=1, output_d=2):
		self.X = data_set
		self.k = k
		self.N = data_set.shape[0]
		self.d = data_set.shape[1]
		self.hidden_d = hidden_node_count
		self.output_d = output_d
		self.run_name = run_name

		self.original_cost = 0
		self.final_cost = 0

		self.U_matrix = np.random.random([self.N,self.k])
		self.H_matrix = np.eye(self.N) - np.ones((self.N,self.N))/self.N
		self.I = np.eye(self.N)

		self.lambdaV = -100
		self.alpha = 0.001
	
		self.dtype = torch.FloatTensor
		self.xTor = torch.from_numpy(data_set)
		self.xTor = Variable(self.xTor.type(self.dtype), requires_grad=False)


		# Random Fourier Features	
		self.sigma = sigma
		self.sample_num = 20000

		b = 2*np.pi*np.random.rand(1, self.sample_num)
		b = np.matlib.repmat(b, self.N, 1)

		self.phase_shift = torch.from_numpy(b)
		self.phase_shift = Variable(self.phase_shift.type(self.dtype), requires_grad=False)

		u = np.random.randn(self.output_d, self.sample_num)/(self.sigma)
		self.rand_proj = torch.from_numpy(u)
		self.rand_proj = Variable(self.rand_proj.type(self.dtype), requires_grad=False)

		u2 = np.random.randn(self.d, self.sample_num)/(self.sigma)
		self.rand_proj2 = torch.from_numpy(u2)
		self.rand_proj2 = Variable(self.rand_proj2.type(self.dtype), requires_grad=False)

		self.RBF_method = 'RFF'


		np.set_printoptions(precision=3)
		np.set_printoptions(threshold=np.nan)
		np.set_printoptions(linewidth=300)
		np.set_printoptions(suppress=True)


		Vgamma = 1/(2*self.sigma*self.sigma)
		self.init_K = sklearn.metrics.pairwise.rbf_kernel(self.X, gamma=Vgamma)
		self.init_K = torch.from_numpy(self.init_K)
		self.init_K = Variable(self.init_K.type(self.dtype), requires_grad=False)



	def plot_clustering(self, X=None, allocation=[]):
		if X == None: X = self.X

		plt.figure(1)
		plt.subplot(111)
		plt.title('moon')
		
		if len(allocation) != 0:
			idx = np.unique(allocation)
			for mm in idx:
				subgroup = X[allocation == mm]
				plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
		else:
			plt.plot(X[:,0], X[:,1], color=colors.keys()[0] , marker='o', linestyle='None')

		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('Alternative Clustering')
		
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
		plt.show()

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

	def initial_pass(self):
		Y = self.NN(self.xTor)
		P = torch.cos(torch.mm(Y,self.rand_proj) + self.phase_shift)
		K = torch.mm(P, P.transpose(0,1))
		K = (2.0/self.sample_num)*K
		#K = torch.clamp(K, 0)	#clamp doesn't seem to do back prop

		diff = self.init_K - K
		error = (diff*diff).sum()
		return error

	def minimize_initial_error(self):
	
		learning_rate = 1

		while True:
			cost = self.initial_pass()
			
			self.NN.zero_grad()
			cost.backward()
		
			while True:		#	Adaptive Learning Rate
				for param in self.NN.parameters():
					param.data -= learning_rate * param.grad.data
	
				new_cost = self.initial_pass()

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
			if (np.absolute(new_cost.data.numpy() - cost.data.numpy()))/np.absolute(new_cost.data.numpy()) < 0.00001: print('Cost Exit'); break;
			if learning_rate < 0.0000001: print('Learning Rate Exit'); break

		import pdb; pdb.set_trace()


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

	def compute_Gaussian_Laplacian(self, input_data, RBF_method='RFF'):
		#input_data = input_data/torch.unsqueeze(torch.sqrt((input_data*input_data).sum(1)),1)		#row normalized

		if RBF_method == 'sklearn':
			Vgamma = 1/(2*self.sigma*self.sigma)
			K = sklearn.metrics.pairwise.rbf_kernel(input_data.data.numpy(), gamma=Vgamma)

			K = torch.from_numpy(K)
			K = Variable(K.type(self.dtype), requires_grad=False)


			#D1 = torch.unsqueeze(torch.sqrt(1/K.sum(1)),1)
			#D = torch.mm(D1, D1.transpose(0,1))	
			#L = K*D
			#L = self.apply_centering(L)												# HDKDH
			#U = self.calc_U(L)
			#Ku = U.dot(U.T)
			#print 'Cost : ' ,  -(Ku*L).sum()											# Tr(UU' HDKDH)

			#U = normalize(U, norm='l2', axis=1)
			#allocation = KMeans(self.k).fit_predict(U)

			#self.plot_clustering(allocation)
			#import pdb; pdb.set_trace()

			#RBF_method = 'RFF'
		if RBF_method == 'RFF': # Use random fourier features
			if input_data.data.numpy().shape[1] == self.rand_proj.data.numpy().shape[0]:
				P = torch.cos(torch.mm(input_data,self.rand_proj) + self.phase_shift)
			else:
				P = torch.cos(torch.mm(input_data,self.rand_proj2) + self.phase_shift)

			K = torch.mm(P, P.transpose(0,1))
			K = (2.0/self.sample_num)*K
			K = K + 0.03
			print 'min K : ', K.min().data.numpy()[0]
			#import pdb; pdb.set_trace()
			#K = torch.clamp(K, 0)	#clamp doesn't seem to do back prop

			#D1 = torch.unsqueeze(torch.sqrt(1/K.sum(1)),1)
			#D = torch.mm(D1, D1.transpose(0,1))	
			#L = K*D
			#L = self.apply_centering(L)												# HDKDH
			#U = self.calc_U(L)
			#Ku = U.dot(U.T)
			#print 'Cost : ' ,  -(Ku*L).sum()											# Tr(UU' HDKDH)

			#U = normalize(U, norm='l2', axis=1)
			#allocation = KMeans(self.k).fit_predict(U)

			#self.plot_clustering(allocation)
			#import pdb; pdb.set_trace()

		#RBF_method = 'element wise'
		if RBF_method == 'element wise': # Use actual gaussian kernel
			K = torch.FloatTensor(self.N, self.N)
			K = Variable(K.type(self.dtype), requires_grad=False)
			Y = input_data
			
			#Y = Y/torch.unsqueeze(torch.sqrt((Y*Y).sum(1)),1)		#row normalized
	
			for i in range(self.N):
				for j in range(self.N):
					tmpY = (Y[i,:] - Y[j,:]).unsqueeze(0)
					eVal = -(torch.mm(tmpY, tmpY.transpose(0,1)))/(2*self.sigma*self.sigma)
					K[i,j] = torch.exp(eVal)

		D1 = torch.unsqueeze(torch.sqrt(1/K.sum(1)),1)
		D = torch.mm(D1, D1.transpose(0,1))	
		L = K*D
		#import pdb; pdb.set_trace()


		return L

	def apply_centering(self, sqMatrix):
		if type(sqMatrix) == type(self.xTor):
			sqMatrix = sqMatrix.data.numpy()

		centered = self.H_matrix.dot(sqMatrix).dot(self.H_matrix)
		centered = (centered + centered.T)/2
		return centered

	def calc_U(self, input_kernel):
		eigenValues,eigenVectors = np.linalg.eigh(input_kernel)
		
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
		phi = self.apply_centering(phi)							# HUU'H
		
		phi = phi.astype(np.float32)
	
		phi = torch.from_numpy(phi)
		phi = Variable(phi.type(self.dtype), requires_grad=False)
		return phi

	def forward_pass(self, phi):		#	Gaussian Version
		Y = self.NN(self.xTor)
	
		#L = self.compute_Linear_Laplacian(Y)
		L = self.compute_Gaussian_Laplacian(Y, RBF_method=self.RBF_method)
		
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
				if (np.absolute(new_cost.data.numpy() - cost.data.numpy()))/np.absolute(new_cost.data.numpy()) < 0.001: print('Cost Exit'); break;
				if learning_rate < 0.0000001: print('Learning Rate Exit'); break


				#Y = self.NN(self.xTor)
				#new_sigma = np.median(sklearn.metrics.pairwise.pairwise_distances(Y.data.numpy()))
				#print 'sigma : ' , new_sigma
				#L = self.compute_Gaussian_Laplacian(Y)
				#self.draw_heatMap(L)
				#import pdb; pdb.set_trace()

		Y = self.NN(self.xTor)
		L = self.compute_Gaussian_Laplacian(Y, RBF_method=self.RBF_method)
		
		return L	

	def run(self):
		L = self.compute_Gaussian_Laplacian(self.xTor,RBF_method='sklearn')			#  'element wise' sklearn, RFF, DKD
		L = self.apply_centering(L)												# HDKDH
		U = self.calc_U(L) #.data.numpy())
		
		Ku = U.dot(U.T)
		self.original_cost = -(Ku*L).sum()											# Tr(UU' HDKDH)
		print 'Original cost : ' , self.original_cost

		L = self.update_W(U)
		L = self.apply_centering(L)
		U = self.calc_U(L)

		Ku = U.dot(U.T)
		self.final_cost = -(Ku*L).sum()
		print 'Final cost : ' , self.final_cost

		U = normalize(U, norm='l2', axis=1)
		allocation = KMeans(self.k).fit_predict(U)
		
		return allocation


