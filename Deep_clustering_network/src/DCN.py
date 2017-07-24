
import sklearn.metrics
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

class DCN:
	def __init__(self, data_set, k):
		self.X = data_set
		self.k = k
		self.N = data_set.shape[0]
		self.d = data_set.shape[1]
		self.hidden_d = self.d + 1					# hidden layer has 1 extra dimension
		self.output_d = k							# output layer has k dimensions
		self.lambdaV = 0.01
		self.I = np.eye(self.N)
		self.mini_batch_size = 40

		self.loop = True
		self.H_matrix = np.eye(self.N) - np.ones((self.N,self.N))/self.N
		self.U_matrix = np.random.random([self.N,self.k])
		self.change_in_U = 1000
		self.small_enough_U = 0.001
		self.max_U_loop = 100
		self.U_loop_counter = 0

		#ptorch 
		self.dtype = torch.FloatTensor
		#self.dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU	
		self.NN = torch.nn.Sequential(
			torch.nn.Linear(self.d, self.hidden_d),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.output_d),
		)
		self.NN_hold = torch.nn.Sequential(
			torch.nn.Linear(self.d, self.hidden_d),
			torch.nn.ReLU(),
			torch.nn.Linear(self.hidden_d, self.output_d),
		)


		np.set_printoptions(precision=2)
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

	def create_Phi(self, x_i_list, x_j_list):
		H = self.H_matrix
		l = self.lambdaV

		Ku = self.U_matrix.dot(self.U_matrix.T)			# kernel U
		Phi_large = self.I - l*H.dot(Ku).dot(H)			

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
			self.kernel = sklearn.metrics.pairwise.rbf_kernel(kernel_input, gamma=1)
		
		if add_scaling:
			self.D_matrix = np.diag(1/np.sqrt(np.sum(self.kernel,axis=1))) # 1/sqrt(D)
			self.L = self.D_matrix.dot(self.kernel).dot(self.D_matrix)
		else:
			self.L = self.kernel
		
		self.L_centered = self.H_matrix.dot(self.L).dot(self.H_matrix)

	def init_W(self):
		self.w1 = Variable(torch.randn(self.d, self.hidden_d).type(self.dtype), requires_grad=True)
		self.w2 = Variable(torch.randn(self.hidden_d, self.output_d).type(self.dtype), requires_grad=True)

	def calc_U(self):
		eigenValues,eigenVectors = np.linalg.eigh(self.L_centered)
	
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
		[y_i, y_j, cost] = self.forward_pass(x_i, x_j, phi)

		self.calc_Kernel('linear', y_i.data.numpy())

	def forward_pass(self, x_i, x_j, phi):
		y_i = self.NN(x_i)
		y_j = self.NN(x_j)

		cost = (phi*torch.mm(y_i,y_j.transpose(0,1))).sum()
		return [y_i, y_j, cost]


	def update_W(self):
		learning_rate = 0.01


		while True:
			[xi_idx, xj_idx, x_i, x_j] = self.create_miniBatch(self.X, self.mini_batch_size)	# x is sub batch of data, u is the corresponding clustering
			phi = self.create_Phi(xi_idx, xj_idx)	
			[y_i, y_j, cost] = self.forward_pass(x_i, x_j, phi)
			self.NN.zero_grad()
			cost.backward()

			while True:		#	Adaptive Learning Rate
				for param in self.NN.parameters():
					param.data -= learning_rate * param.grad.data

				[new_y_i, new_y_j, new_cost] = self.forward_pass(x_i, x_j, phi)
				if(new_cost.data[0] > cost.data[0]): # if got worse, undo and lower the learning rate. 
					for param in self.NN.parameters():
						param.data += learning_rate * param.grad.data

					learning_rate = learning_rate*0.6
				else: 
					break
				if learning_rate < 0.000000001: break


			grad_norm = 0	
			for param in self.NN.parameters():
				grad_norm += param.grad.data.norm()
			if grad_norm < 0.001: break
			print(learning_rate, ' , ' , cost.data[0], ' , ' , grad_norm)


	
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

	def run(self):
		self.calc_Kernel('RBK', self.X, True)
		self.init_W()
		self.calc_U()
		print self.get_clustering_results()

		while(self.loop):
			self.update_W()
			self.update_K()
			self.calc_U()
			self.loop = self.check_convergence()
			print '\n-----------\n\n'

		#import pdb; pdb.set_trace()
		self.loop = True

		return self.get_clustering_results()
