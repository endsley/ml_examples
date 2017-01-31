#!/usr/bin/python

import numpy as np
import pdb
import csv
import sklearn.metrics
from create_y_tilde import *
from create_gamma_ij import *
from StringIO import StringIO
from scipy.optimize import minimize
import time 

#import autograd.numpy as np
#from autograd import grad

## Define a function Tr(WTA W), we know that gradient = (A+AT)W
#def cost_foo(W, db): 
#	K = db['Kernel_matrix']
#	U = db['U_matrix']
#	H = db['H_matrix']
#	D = db['D_matrix']
#	l = db['lambda']
#	Y = db['Y_matrix']
#
#	s1 = np.dot(np.dot(D,H),U)
#	HSIC_1 = np.dot(s1, np.transpose(s1))*K
#
#	s2 = np.dot(np.dot(D,H),Y)
#	HSIC_2 = l*np.dot(s2, np.transpose(s2))*K
#
#	return np.sum(HSIC_2 - HSIC_1)
#
#grad_foo = grad(cost_foo)       # Obtain its gradient function

class cost_function:
	def __init__(self, db):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]

		self.iv = np.array(range(self.N))
		self.jv = self.iv

		self.sigma2 = np.power(db['sigma'],2)

		self.y_tilde = None
		self.W = None
		self.gamma = np.zeros((self.N, self.N))
		self.exp = np.zeros((self.N, self.N))
		self.gamma_exp = np.empty((self.N, self.N))
		self.A_memory_feasible = True

		self.psi = None			# This is the middle term that doesn't change unless U or Y update
		self.Q = None			# This is the tensor term of ( X tensor 1 ) - (1 tensor X)

		#try:
		#	self.A = np.empty((self.N,self.N,self.d, self.d))
		#	self.Aw = np.empty((self.N,self.N,self.d, self.q))
		#	self.create_A()
		#except:
		#	self.A_memory_feasible = False
		#	raise
	
		self.calc_Q()

	def calc_Q(self):
		try:
			X = self.db['data']
			one_vector = np.ones((self.N,1))
			self.Q = np.kron(X,one_vector) - np.kron(one_vector, X)
		except:
			self.A_memory_feasible = False
	
	def calc_psi(self, Y_columns=None): # psi = H(UU'-l YY')H
		db = self.db
		Y = db['Y_matrix']

		if(Y_columns != None): 
			Y = Y[:,0:Y_columns]
		U = db['U_matrix']
		H = db['H_matrix']

		self.psi = H.dot(U.dot(U.T) - db['lambda']*Y.dot(Y.T)).dot(H)
		return self.psi

	def create_D_matrix(self, kernel):
		d_matrix = np.diag(1/np.sqrt(np.sum(kernel,axis=1))) # 1/sqrt(D)
		return d_matrix




	def create_Kernel(self, W):
		db = self.db
		sigma = db['sigma']
		X = db['data'].dot(W)
		gamma_V = 1.0/(2*np.power(sigma,2))
		db['Kernel_matrix'] = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma_V)
		db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

		return db['Kernel_matrix']


	def derivative_test(self, db):
		W = db['W_matrix']
		K = self.create_Kernel(W)
		H = db['H_matrix']
		U = db['U_matrix']

		K = self.create_Kernel(W)
		D = np.diag(db['D_matrix'])
		DD = np.outer(D,D)
		const_matrix = DD*self.psi*K

		A = np.zeros((self.d, self.d))
		for i in self.iv:
			for j in self.jv:
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
				A_ij = np.dot(x_dif.T, x_dif)
				A += const_matrix[i,j]*A_ij

		if True:# Use eig
			[S2,U2] = np.linalg.eigh(A)
			eigsValues = S2[0:db['q']]
			#W = U2[:,0:db['q']]
		else:
			# Use svd
			[U,S,V] = np.linalg.svd(A)
			reverse_S = S[::-1]
			eigsValues = reverse_S[0:db['q']]
			#W = np.fliplr(U)[:,0:db['q']]

		new_gradient = A.dot(W)
		Lagrange_gradient = new_gradient - W*eigsValues
		new_gradient_mag = np.linalg.norm(Lagrange_gradient)
		return new_gradient_mag

	def cluster_quality(self, db):
		W = db['W_matrix']
		K = self.create_Kernel(W)
		D = db['D_matrix']
		H = db['H_matrix']
		U = db['U_matrix']

		return np.trace( H.dot(D).dot(K).dot(D).dot(H).dot(U).dot(U.T) ) 

	def alternative_quality(self, db):
		W = db['W_matrix']
		K = self.create_Kernel(W)
		D = db['D_matrix']
		H = db['H_matrix']
		Y = db['Y_matrix']

		return np.trace( H.dot(D).dot(K).dot(D).dot(H).dot(Y).dot(Y.T) ) 

	def calc_cost_function(self, W, also_calc_Phi=False, update_D_matrix=False, Y_columns=None): #Phi = the matrix we perform SVD on
		db = self.db
		if type(self.psi) == type(None): self.calc_psi(Y_columns)

		#start_time = time.time() 
		K = self.create_Kernel(W)
		D = np.diag(db['D_matrix'])
		DD = np.outer(D,D)

		const_matrix = DD*self.psi*K
		cost = -np.sum(const_matrix)


		#print(cost)
		#print("--- 1 : %s seconds ---" % (time.time() - start_time))


		#start_time = time.time() 
		#depr_cost = self.calc_cost_function_depre(W)
		#print(depr_cost)
		#print("--- 2 : %s seconds ---" % (time.time() - start_time))
		#import pdb; pdb.set_trace()

		if not also_calc_Phi: return cost
		if self.A_memory_feasible:
			O = np.reshape(const_matrix, (1,const_matrix.size))
			Phi = ((self.Q.T*O).dot(self.Q))/self.sigma2
			return [cost, Phi]
		else:
			print '\n\nYou still need to write the part where memory is not feasible.\n\n'
			#	You will have to multiply each A and add them up
			raise


#
#	def create_gamma_exp_A(self, W):
#		self.create_y_tilde()
#		self.create_gamma()
#
#		gamma_exp = self.create_gamma_exps(W)
#		matrix_sum = np.zeros((self.d, self.d))
#		for i in self.iv:
#			for j in self.jv:
#				A = self.get_A(i, j)
#				matrix_sum += gamma_exp[i][j]*A
#		
#		matrix_sum = matrix_sum/float(self.sigma2)
#		return matrix_sum
#
#	def create_gamma_exps(self, W):
#		exp_wAw = self.create_Kernel(W)
#		self.gamma_exp = self.gamma*exp_wAw
#		return self.gamma_exp
#
#	def create_y_tilde(self):
#		#	tr(H K_y H D K D)
#		#	tr((D H K_y H D) K)
#		#	tr(M K)
#
#		db = self.db
#		Y = db['Y_matrix']
#		H = db['H_matrix']
#		D = db['D_matrix']
#
#		K_y = Y.dot(Y.T)
#		inner_p = H.dot(K_y).dot(H)
#		db['y_tilde'] = D.dot(inner_p).dot(D)
#
#	def create_gamma(self):
#		db = self.db
#		yt = db['y_tilde']
#		U = db['H_matrix'].dot(db['U_matrix'])
#
#		for i in self.iv:
#			for j in self.jv:
#				degree_of_vertex = np.diag(db['D_matrix'])
#				ith_row = U[i,:]
#				jth_row = U[j,:]
#			
#				u_dot = np.dot(ith_row,jth_row)
#				part_1 = u_dot*degree_of_vertex[i]*degree_of_vertex[j]
#			
#				self.gamma[i][j] = part_1 - db['lambda'] * yt[i,j]


#	Deprecated version
#	def create_Kernel_depr(self, W):
#		db = self.db
#		self.create_Aw(W)
#		kernel = np.zeros((db['N'], db['N']))
#
#		for i in self.iv:
#			for j in self.jv:
#				#print i, j
#				Aw = self.get_Aw(i,j, W)
#				kernel[i][j] = np.exp(np.sum(-W*Aw)/(2*self.sigma2))
#
#		return kernel


#	This function is now deprecated
#	def initialize_constants(self):
#		self.create_y_tilde()
#		self.create_gamma()

##	Deprecated version
#	def calc_cost_function_depre(self, W):
#		self.initialize_constants()
#		self.create_gamma_exps(W)
#		return np.sum(self.gamma_exp)

#	def get_A(self, i, j):
#		if self.A_memory_feasible:
#			return self.A[i][j]
#		else:
#			x_dif = self.db['data'][i] - self.db['data'][j]
#			x_dif = x_dif[np.newaxis]
#			return np.dot(x_dif.T, x_dif)
#		
#
#	def create_A(self):
#		db = self.db
#		for i in self.iv:
#			for j in self.jv:
#				x_dif = db['data'][i] - db['data'][j]
#				x_dif = x_dif[np.newaxis]
#				self.A[i][j] = np.dot(x_dif.T, x_dif)
#
#	def get_Aw(self, i, j, W):
#		if self.A_memory_feasible: 
#			return self.Aw[i][j]
#		else:
#			A = self.get_A(i,j)
#			return A.dot(W)
#
#	def create_Aw(self,W):
#		if not self.A_memory_feasible: return
#
#		db = self.db
#		for i in self.iv:
#			for j in self.jv:
#				A = self.get_A(i,j)
#				self.Aw[i][j] = A.dot(W)


def test_1():		# optimal = 2.4309
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=np.nan)
	np.set_printoptions(linewidth=300)


	db = {}
	db['data'] = np.array([[5,5],[4,4],[-5,5],[-4,4],[5,-4],[4,-3],[-5,-4],[-4,-3]])
	db['W_matrix'] = np.array([[1],[1]])
	db['sigma'] = 2
	db['lambda'] = 1
	db['N'] = db['data'].shape[0]
	N = float(db['N'])
	db['C_num'] = 2

	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")
	db['Kernel_matrix'] = sklearn.metrics.pairwise.rbf_kernel(db['data'], gamma=(0.5/np.power(db['sigma'],2)))
	db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)
	L = db['D_matrix'].dot(db['Kernel_matrix']).dot(db['D_matrix'])
	[U,S,V] = np.linalg.svd(L)
	db['U_matrix'] = U[:,:db['C_num']]

	db['Y_matrix'] = np.array([[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1]])
	db['H_matrix'] = np.eye(db['N']) - (1.0/N)*np.ones((db['N'], db['N']))

	c_f = cost_function(db)
	print c_f.calc_cost_function(db['W_matrix'])

	#pdb.set_trace()

#test_1()
