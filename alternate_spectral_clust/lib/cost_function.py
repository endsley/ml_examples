#!/usr/bin/python

import numpy as np
import pdb
import csv
import sklearn.metrics
from create_y_tilde import *
from create_gamma_ij import *
from StringIO import StringIO
from scipy.optimize import minimize


class cost_function:
	def __init__(self, db):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]
		self.q = db['W_matrix'].shape[1]

		self.iv = np.array(range(self.N))
		self.jv = self.iv

		self.sigma2 = np.power(db['sigma'],2)

		self.y_tilde = None
		self.W = None
		self.gamma = np.zeros((self.N, self.N))
		self.exp = np.zeros((self.N, self.N))
		self.A = np.empty((self.N,self.N,self.d, self.d))
		self.Aw = np.empty((self.N,self.N,self.d, self.q))
		self.last_used_W = np.empty((self.d, self.q))
		self.gamma_exp = np.empty((self.N, self.N))

		self.create_A()


	def initialize_constants(self):
		self.create_y_tilde()
		self.create_gamma()

	def create_y_tilde(self):
		#	tr(H K_y H D K D)
		#	tr((D H K_y H D) K)
		#	tr(M K)

		db = self.db
		Y = db['Y_matrix']
		H = db['H_matrix']
		D = db['D_matrix']

		K_y = Y.dot(Y.T)
		inner_p = H.dot(K_y).dot(H)
		db['y_tilde'] = D.dot(inner_p).dot(D)

	def create_gamma(self):
		db = self.db
		yt = db['y_tilde']

		for i in self.iv:
			for j in self.jv:
				degree_of_vertex = np.diag(db['D_matrix'])
				ith_row = db['U_matrix'][i,:]
				jth_row = db['U_matrix'][j,:]
			
				u_dot = np.dot(ith_row,jth_row)
				part_1 = u_dot*degree_of_vertex[i]*degree_of_vertex[j]
			
				self.gamma[i][j] = part_1 - db['lambda'] * yt[i,j]

	def create_A(self):
		db = self.db
		for i in self.iv:
			for j in self.jv:
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
				self.A[i][j] = np.dot(x_dif.T, x_dif)

	def create_Aw(self,W):
		db = self.db
		for i in self.iv:
			for j in self.jv:
				self.Aw[i][j] = self.A[i][j].dot(W)

	def create_D_matrix(self, kernel):
		d_matrix = np.diag(1/np.sqrt(np.sum(kernel,axis=1))) # 1/sqrt(D)
		return d_matrix

	def create_Kernel(self, W):
		db = self.db
		self.create_Aw(W)
		kernel = np.zeros((db['N'], db['N']))

		for i in self.iv:
			for j in self.jv:
				kernel[i][j] = np.exp(np.sum(-W*self.Aw[i][j])/(2*self.sigma2))

		return kernel

	def create_gamma_exps(self, W):
		exp_wAw = self.create_Kernel(W)
		self.gamma_exp = self.gamma*exp_wAw
		return self.gamma_exp

	def create_gamma_exp_A(self, W):
		gamma_exp = self.create_gamma_exps(W)
		matrix_sum = np.zeros((self.d, self.d))
		for i in self.iv:
			for j in self.jv:
				matrix_sum += gamma_exp[i][j]*self.A[i][j]

		return matrix_sum

	def calc_cost_function(self, W):
		self.create_gamma_exps(W)
		return np.sum(self.gamma_exp)

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
