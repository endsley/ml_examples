#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
from create_gamma_ij import *
from StringIO import StringIO
from sklearn.preprocessing import normalize
import pdb


class exponential_solver:
	def __init__(self, db, iv, jv, y_tilde):
		self.db = db
		self.iv = iv
		self.jv = jv
		self.zi = 0
		self.zj = 0

		self.optimal_val = 0
		self.current_cost = 0
		self.gamma_array = 0

		self.W_shape = db['W_matrix'].shape
		self.wi = self.W_shape[0]
		self.wj = self.W_shape[1]

		self.matrix_gap = 0		# measures the distance between W , Z matrix
		self.last_matrix_gap = 0
		self.learning_rate = 1
		self.y_tilde = y_tilde


	def create_gamma_ij(self, db, i, j):
		if type(self.gamma_array) == type(0):
			return create_gamma_ij(db, self.y_tilde, i, j)
		else:
			return self.gamma_array[i,j]

	def create_A_ij_matrix(i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(np.transpose(x_dif), x_dif)

	def Lagrange_W(self, W):
		db = self.db
		Z = db['Z_matrix']
		L1 = db['L1']
		L2 = db['L2']
		Wsh = self.W_shape

		W2 = W.reshape(Wsh)
		I = np.eye(self.wj)
	
		#	Setting up the cost function
		cost_foo = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(db, i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

		self.current_cost = cost_foo
		#print self.current_cost
		term1 = W2.T.dot(Z) - I
		term2 =  W2 - Z 	

		Lagrange = np.trace(L1.dot(term1)) + np.trace(L2.dot(term2))
		Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)
		foo = cost_foo + Lagrange + Aug_lag
		#print cost_foo, foo 
		return foo

	def calc_lagrange_results(self, L):
		W = self.db['W_matrix']
		Z2 = self.db['Z_matrix']
		Z_shape = Z2.shape
		eye_matrix = np.eye(Z_shape[1])

		L1 = L[0:self.zj,:]
		L2 = L[self.zj:,:].T


		#	Setting up the cost function
		cost_foo = 0

		Lagrange = np.trace(L1.dot(W.T.dot(Z2) - eye_matrix)) + np.sum(L2.T*(W - Z2))
		term1 = ( Z2.T.dot(W) - eye_matrix )
		term2 =  W - Z2 	
		Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)

		foo = cost_foo + Lagrange + Aug_lag
		return foo

	def Lagrange_Z(self, Z):
		W = self.db['W_matrix']
		L1 = self.db['L1']
		L2 = self.db['L2']

		Z_shape = self.db['Z_matrix'].shape
		Z2 = Z.reshape(Z_shape)

		I = np.eye(Z_shape[1])
	
		#	Setting up the cost function
		cost_foo = 0

		term1 = ( W.T.dot(Z2) - I)
		term2 =  W - Z2 	
		Lagrange = np.trace(L1.dot(term1)) + np.sum(L2.dot(term2))
		Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)

		foo = cost_foo + Lagrange + Aug_lag
		return foo

	def calc_dual(self):
		db = self.db
		Z = db['Z_matrix']
		W = db['W_matrix']

		A = np.append(Z.T, np.eye( self.zi ), axis=0)
		B = np.append(np.zeros(Z.T.shape), np.eye( self.zi ), axis=0)
		C = np.append(np.eye(self.zj), np.zeros(Z.shape), axis=0)

		if self.matrix_gap > self.last_matrix_gap:
			self.learning_rate = self.learning_rate*0.9

		self.last_matrix_gap = self.matrix_gap

		db['L'] = db['L'] + self.learning_rate*(A.dot(W)-B.dot(Z)-C)
		db['L1'] = db['L'][0:self.zj,:]
		db['L2'] = db['L'][self.zj:,:].T

		#L = L + alpha*(A.dot(W)-B.dot(Z)-C)
		#L1 = db['L'][0:zj,:]
		#L2 = db['L'][zj:,:].T
		#return {'L':L, 'L1':L1, 'L2':L2}


	def w_optimize(self):
		#	Calculate dL/dw gradient
		db = self.db
		W = db['W_matrix']
		Z = db['Z_matrix']
		L1 = db['L1']
		L2 = db['L2']

		#	Calc Base
		dL_dW_1 = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(db, 0, i, j)
				dL_dW_1 = dL_dW_1 + (x_dif.T.dot(x_dif).dot(W))* gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

		dL_dW_2 = Z.dot(L1) + L2.T + 2*Z.dot(Z.T.dot(W) - np.identity(db['q']))

	def run(self):	
		db = self.db

		self.zi = db['Z_matrix'].shape[0]
		self.zj = db['Z_matrix'].shape[1]

		loop_count = 0
		stay_in_loop = True

		while stay_in_loop: #nelder-mead, BFGS, CG , maxiter, Newton-CG
			result_w = minimize(self.Lagrange_W, db['W_matrix'], method='nelder-mead', options={'xtol': 1e-4, 'disp': False})
			db['W_matrix'] = result_w.x.reshape(db['W_matrix'].shape)

			result_z = minimize(self.Lagrange_Z, db['Z_matrix'], method='nelder-mead', options={'xtol': 1e-4, 'disp': False})
			db['Z_matrix'] = result_z.x.reshape(db['Z_matrix'].shape)
			#db['Z_matrix'] = normalize(db['Z_matrix'], norm='l2', axis=0)

			#self.w_optimize()
			self.calc_dual()

			loop_count += 1
			self.matrix_gap = np.abs(np.sum(db['W_matrix'].T.dot(db['Z_matrix']) - np.eye(self.zj)))
			#print loop_count, self.matrix_gap, self.learning_rate, 'cost : ' , self.current_cost

			if self.matrix_gap < 0.001: 
				print('Exit base on threshold')
				stay_in_loop = False
			if loop_count > 200: 
				print('Exit base on loop_count')
				stay_in_loop = False

		self.optimal_val = result_w.fun	
		return result_w





def test_1():		# optimal = 2.4309
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]])
	db['W_matrix'] = np.array([[1,10],[1,1],[0,0]])
	
	db['L1'] = np.array([[1,0], [0,2]])		# q x q
	db['L2'] = np.array([[2,3,1],[0,0,1]])	# q x d
	db['L'] = np.append(db['L1'], db['L2'].T, axis=0)
	
	
	iv = np.array([0])
	jv = np.array([1,2])
	esolver = exponential_solver(db, iv, jv,0)
	esolver.gamma_array = np.array([[0,1,2,-1]])
	esolver.run()
	
	esolver.Lagrange_W(db['W_matrix'])
	print esolver.current_cost
	pdb.set_trace()


#test_1()
