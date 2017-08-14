#!/usr/bin/python

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


class DCN:
	def __init__(self, data_set, k, run_name):
		self.X = data_set
		self.k = k
		self.N = data_set.shape[0]
		self.d = data_set.shape[1]
		self.hidden_d = self.d + 1000					# hidden layer has 1 extra dimension
		self.output_d = k							# output layer has k dimensions
		self.lambdaV = -100
		self.alpha = 0.001
		self.I = np.eye(self.N)
		self.mini_batch_size = 40
		self.run_name = run_name

		self.loop = True
		self.H_matrix = np.eye(self.N) - np.ones((self.N,self.N))/self.N
		self.U_matrix = np.random.random([self.N,self.k])
		self.change_in_U = 1000
		self.small_enough_U = 0.001
		self.max_U_loop = 100
		self.U_loop_counter = 0

		d_matrix = sklearn.metrics.pairwise.pairwise_distances(data_set, Y=None, metric='euclidean')
		sigma = np.median(d_matrix)
		self.gamma = 1/(2*np.power(sigma,2))
		
		

		#ptorch 
		self.dtype = torch.FloatTensor
		#self.dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU	
		self.NN = torch.nn.Sequential(
			torch.nn.Linear(self.d, self.hidden_d, bias=True),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.hidden_d, bias=True),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.hidden_d, bias=True),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.hidden_d, bias=True),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.output_d, bias=True),
			torch.nn.Sigmoid(),
		)
		self.xTor = torch.from_numpy(data_set)
		self.xTor = Variable(self.xTor.type(self.dtype), requires_grad=False)


		np.set_printoptions(precision=3)
		np.set_printoptions(threshold=np.nan)
		np.set_printoptions(linewidth=300)
		np.set_printoptions(suppress=True)

	#	Auxiliary Functions
	def create_miniBatch(self, X, bsize):		# x_i, x_j is sub batch of data	
		N = X.shape[0]
		if(bsize == N):	# return the original matrix
			x = torch.from_numpy(self.X)
			x = Variable(x.type(self.dtype), requires_grad=False)
			x_list = np.arange(N)

			return [x_list, x_list, x, x]
		else:
			x_i_list = np.random.permutation(N)[0:bsize] 
			x_j_list = np.random.permutation(N)[0:bsize] 

			x_i = np.take(X, x_i_list, axis=0)
			x_j = np.take(X, x_j_list, axis=0)

			x_i = torch.from_numpy(x_i)
			x_i = Variable(x_i.type(self.dtype), requires_grad=False)
			x_j = torch.from_numpy(x_j)
			x_j = Variable(x_j.type(self.dtype), requires_grad=False)

			return [x_i_list, x_j_list, x_i, x_j]

	def transform_data(self):
		return self.NN(self.xTor)
		

	def create_Phi(self, x_i_list, x_j_list):
#		#	Using a rbf as Phi
#		phi = sklearn.metrics.pairwise.rbf_kernel(self.X, gamma=self.gamma)
#		phi = torch.from_numpy(phi)
#		phi = Variable(phi.type(self.dtype), requires_grad=False)
#		return phi


#	#	Using UU^T as the kernel
#		phi = self.U_matrix.dot(self.U_matrix.T)			# kernel U
#		phi = torch.from_numpy(phi)
#		phi = Variable(phi.type(self.dtype), requires_grad=False)
#		return phi


	#	Using HUU^TH as the kernel
		H = self.H_matrix
		l = self.lambdaV

		Ku = self.U_matrix.dot(self.U_matrix.T)			# kernel U
		#Phi_large = l*self.I + H.dot(Ku).dot(H)			
		Phi_large = H.dot(Ku).dot(H)			

		if(len(x_i_list) == self.N): 
			Phi_large = torch.from_numpy(Phi_large)
			Phi_large = Variable(Phi_large.type(self.dtype), requires_grad=False)
			return Phi_large

		phi = np.zeros([self.mini_batch_size,self.mini_batch_size])
		for i in range(len(x_i_list)):
			for j in range(len(x_j_list)):
				phi[i,j] = Phi_large[x_i_list[i], x_j_list[j]]

		phi = torch.from_numpy(phi)
		phi = Variable(phi.type(self.dtype), requires_grad=False)
		return phi
		

	def calc_Kernel(self, kernel_type, kernel_input, add_scaling=False):
		if kernel_type == 'linear':
			self.kernel = kernel_input.dot(kernel_input.T)
		elif kernel_type == 'RBK':
			self.kernel = sklearn.metrics.pairwise.rbf_kernel(kernel_input, gamma=self.gamma)
		
		if add_scaling:
			self.D_matrix = np.diag(1/np.sqrt(np.sum(self.kernel,axis=1))) # 1/sqrt(D)
			self.L = self.D_matrix.dot(self.kernel).dot(self.D_matrix)
		else:
			self.L = self.kernel
		
		self.L_centered = self.H_matrix.dot(self.L).dot(self.H_matrix)

	def calc_cost(self, kernel):
		[xi_idx, xj_idx, x_i, x_j] = self.create_miniBatch(self.X, self.N)	
		phi = self.create_Phi(xi_idx, xj_idx)	

		kernel = torch.from_numpy(kernel)
		kernel = Variable(kernel.type(self.dtype), requires_grad=False)

		cost = (phi*kernel).sum()
		return cost

	def calc_U(self):
		eigenValues,eigenVectors = np.linalg.eigh(self.kernel)
	
		idx = eigenValues.argsort()
		idx = idx[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]

		previous_U = np.copy(self.U_matrix)
		self.U_matrix = eigenVectors[:,:self.k]
		self.change_in_U = np.linalg.norm(previous_U - self.U_matrix)/np.linalg.norm(previous_U)

	#	Main Functions
	def update_K(self):						#	update the kernel(i,j) value to y_i.T.dot(y_j)
		[xi_idx, xj_idx, x_i, x_j] = self.create_miniBatch(self.X, self.N)	
		phi = self.create_Phi(xi_idx, xj_idx)	

		y_i = self.NN(x_i)
		#cost = (phi*torch.mm(y_i,y_i.transpose(0,1))).sum()

		self.calc_Kernel('linear', y_i.data.numpy())

	def forward_pass(self, phi, lmda):
		Y = self.NN(self.xTor)
	
		reg_error = torch.abs(Y).sum(1) - 1
		reg_error = reg_error*reg_error
		regularizer = torch.mm(reg_error.transpose(0,1), lmda)

		K = torch.FloatTensor(self.N, self.N)
		K = Variable(K.type(self.dtype), requires_grad=False)

		start_time = time.time() 
		print 'Before kernel'
		for i in range(self.N):
			for j in range(self.N):
				tmpY = (Y[i,:] - Y[j,:]).unsqueeze(0)
				eVal = -torch.mm(tmpY, tmpY.transpose(0,1))
				K[i,j] = torch.exp(eVal)
		print 'After kernel'
		print("--- %s seconds ---" % (time.time() - start_time))


		cost = -(phi*K).sum() + regularizer
		return [cost, regularizer, reg_error]


	def update_W(self):
		learning_rate = 0.2
		
		for lmda_count in range(10):
			while True:
				[xi_idx, xj_idx, x_i, x_j] = self.create_miniBatch(self.X, self.N)	# x is sub batch of data, u is the corresponding clustering
				phi = self.create_Phi(xi_idx, xj_idx)	
	
	
				lmda = torch.from_numpy(self.lmda_hold)
				lmda = Variable(lmda.type(self.dtype), requires_grad=False)
	
				[cost, regularizer, reg_error] = self.forward_pass(phi, lmda)
				
				self.NN.zero_grad()
				cost.backward()
	
				while True:		#	Adaptive Learning Rate
					for param in self.NN.parameters():
						param.data -= learning_rate * param.grad.data
	
	
					[new_cost, regularizer, reg_error] = self.forward_pass(phi, lmda)
					if(new_cost.data[0] > cost.data[0]): # if got worse, undo and lower the learning rate. 
						for param in self.NN.parameters():
							param.data += learning_rate * param.grad.data
	
						learning_rate = learning_rate*0.6
					else: 
						learning_rate = learning_rate*1.001
						break
					if learning_rate < 0.00000001: break
	
	
				grad_norm = 0	
				for param in self.NN.parameters():
					grad_norm += param.grad.data.norm()
	
				print(learning_rate, ' , ' , cost.data[0], ' , ' , grad_norm, ' , ' , regularizer.data.numpy())
				#if grad_norm < 0.001 and np.absolute(L_grad) < 0.001:
				if grad_norm < 0.1:
					break
	
				if (np.absolute(new_cost.data.numpy() - cost.data.numpy()))/np.absolute(new_cost.data.numpy()) < 0.0001:
					break;

			self.lmda_hold += 0.1*reg_error.data.numpy()
			import pdb; pdb.set_trace()


		Y = self.NN(self.xTor)
		self.kernel = torch.mm(Y,Y.transpose(0,1))
		self.kernel = self.kernel.data.numpy()

		return Y.data.numpy()
	

	def print_weights(self):	
		for param in self.NN.parameters():
			print param.data

	def check_convergence(self):
		self.U_loop_counter += 1
	
		if self.U_loop_counter > self.max_U_loop:	
			return False		# Exit an infinite loop

		if(self.change_in_U < self.small_enough_U):
			return False		

		return True
	
	def get_clustering_results(self):
		self.U_matrix = normalize(self.U_matrix, norm='l2', axis=1)
		self.allocation = KMeans(self.k).fit_predict(self.U_matrix)
		return self.allocation

	def initial_cost(self, L, Y, lmda):	# Calculate the initial cost function
		use_linear_kernel = True

		if use_linear_kernel:
			K = torch.mm(Y,Y.transpose(0,1))
		else: #use_gaussian_kernel
			K = torch.FloatTensor(self.N, self.N)
			K = Variable(K.type(self.dtype), requires_grad=False)
	
			for i in range(self.N):
				for j in range(self.N):
					tmpY = (Y[i,:] - Y[j,:]).unsqueeze(0)
					eVal = -torch.mm(tmpY, tmpY.transpose(0,1))
					K[i,j] = torch.exp(eVal)

		error = L - K
		reg_error = torch.abs(Y).sum(1) - 1
		reg_error = reg_error*reg_error
		regularizer = torch.mm(reg_error.transpose(0,1), lmda)
		loss = error.norm() + regularizer
		return loss

	def initialize_W(self):		#	I initialized the weights by setting them to equal to the kernel
		self.kernel = sklearn.metrics.pairwise.rbf_kernel(self.X, gamma=self.gamma)
		L = torch.from_numpy(self.kernel)
		L = Variable(L.type(self.dtype), requires_grad=False)

		lmda_hold = 0.1*np.ones((self.N,1))


		for idx_out in range(10):		#	This is loop for lambda convergence
			learning_rate = 1
			lmda = torch.from_numpy(lmda_hold)
			lmda = Variable(lmda.type(self.dtype), requires_grad=False)


			for idx in range(200):		#	This is loop for W convergence
				Y = self.NN(self.xTor)
				loss = self.initial_cost(L, Y, lmda)

				self.NN.zero_grad()
				loss.backward()
	
				print idx_out, idx, learning_rate, loss.data.numpy()
				while True:		#	Adaptive Learning Rate
					for param in self.NN.parameters():
						param.data -= learning_rate * param.grad.data

					Y = self.NN(self.xTor)
					new_loss = self.initial_cost(L, Y, lmda)
					if(new_loss.data[0] >= loss.data[0]): # if got worse, undo and lower the learning rate. 
						for param in self.NN.parameters():
							param.data += learning_rate * param.grad.data
	
						learning_rate = learning_rate*0.5
					else: 
						#learning_rate = learning_rate*1.1
						break
					if learning_rate < 0.0000001: 
						break
				if learning_rate < 0.000001: break
	

			reg_error = (torch.abs(Y).sum(1) - 1)
			reg_error = reg_error*reg_error
			lmda_hold += 0.001*reg_error.data.numpy()

			print 'lambda error : ' , (Y.sum(1) - 1).sum()
			import pdb; pdb.set_trace()

		#	How similar was the estimation
		self.kernel = torch.mm(Y,Y.transpose(0,1))
		print L 
		print self.kernel
		self.kernel = self.kernel.data.numpy()

		self.save_W(lmda_hold)
		import pdb; pdb.set_trace()


	def calc_clustering_quality(self):
		H = self.H_matrix

		Ku = self.U_matrix.dot(self.U_matrix.T)			# kernel U
		#Phi_large = H.dot(Ku).dot(H)			

		Y = self.NN(self.xTor)
		K = torch.mm(Y,Y.transpose(0,1)).numpy()
		clustering_quality = (Ku*K).sum()
		print 'Clustering Quality : ' , clustering_quality
		return clustering_quality

	def load_W(self):
		self.NN.load_state_dict(torch.load('./trained_models/' + self.run_name + '.pt'))
		self.lmda_hold = torch.load('./trained_models/' + self.run_name + '_lmda_hold.pt')

		Y = self.NN(self.xTor)
		self.kernel = torch.mm(Y,Y.transpose(0,1))
		self.kernel = self.kernel.data.numpy()

	def save_W(self, lmda_hold):
		torch.save(self.NN.state_dict(), './trained_models/' + self.run_name + '.pt')
		torch.save(lmda_hold, './trained_models/' + self.run_name + '_lmda_hold.pt')

	def run(self):
		self.lmda_hold = 0.1*np.ones((self.N,1))

		start_time = time.time() 
		self.kernel = sklearn.metrics.pairwise.rbf_kernel(self.X, gamma=self.gamma)
		print("--- %s seconds ---" % (time.time() - start_time))


		self.kernel = self.H_matrix.dot(self.kernel).dot(self.H_matrix)			
		self.calc_U()
		print self.get_clustering_results()


		Y = self.update_W()
		import pdb; pdb.set_trace()
		self.allocation = KMeans(self.k).fit_predict(Y)
		return self.allocation

#		while(self.loop):
#			self.update_W()
#
#
#			self.calc_U()
#
#			self.loop = self.check_convergence()
#
#			self.calc_clustering_quality()
#			print '\n-----------\n\n'
#
#			#print 'Later cost : ' , self.calc_cost(self.kernel)
#
#			TD = self.transform_data()
#			Y = TD.data.numpy()
#			K = Y.dot(Y.T)
#			Ku = self.U_matrix.dot(self.U_matrix.T)
#			print 'tr(K) : ', np.trace(K)
#			print 'tr(KHKuH) : ', np.trace(K.dot(self.H_matrix).dot(Ku).dot(self.H_matrix))
#
#			import pdb; pdb.set_trace()
#
#		self.loop = True

#		return #self.get_clustering_results()
