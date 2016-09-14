#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
from create_gamma_ij import *

class exponential_solver:
	def __init__(self, db, y_tilde):
		self.db = db
		self.y_tilde = y_tilde
		self.W_result = 0
		self.gamma_array = 0
		self.optimal_val = 0

	def create_gamma_ij(self, db, y_tilde, i, j):
		if type(self.gamma_array) == type(0):
			return create_gamma_ij(db, self.y_tilde, i, j)
		else:
			return self.gamma_array[i,j]

	def create_A_ij_matrix(i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(np.transpose(x_dif), x_dif)

	def Lagrange_W(self, W, iv, jv):
		db = self.db
		Z = db['Z_matrix']
		L1 = db['L1']
		L2 = db['L2']

		W_shape = db['W_matrix'].shape
		W2 = W.reshape(W_shape)
		eye_matrix = np.eye(W_shape[1])
	
		#	Setting up the cost function
		cost_foo = 0
		for i in iv:
			for j in jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(db, self.y_tilde, i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

		self.optimal_val = cost_foo
		Lagrange = np.trace(L1.dot(W2.T.dot(Z) - eye_matrix)) + np.sum(L2.T*(W2 - Z))
		term1 = ( Z.T.dot(W2) - eye_matrix )
		term2 =  W2 - Z 	
		Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)
		foo = cost_foo + Lagrange + Aug_lag
	
		return foo

	def Lagrange_Z(self, Z):
		W = self.db['W_matrix']
		L1 = self.db['L1']
		L2 = self.db['L2']

		Z_shape = self.db['Z_matrix'].shape
		Z2 = Z.reshape(Z_shape)

		eye_matrix = np.eye(Z_shape[1])
	
		#	Setting up the cost function
		cost_foo = 0

		Lagrange = np.trace(L1.dot(W.T.dot(Z2) - eye_matrix)) + np.sum(L2.T*(W - Z2))
		term1 = ( Z2.T.dot(W) - eye_matrix )
		term2 =  W - Z2 	
		Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)

		foo = cost_foo + Lagrange + Aug_lag
		return foo


	def run(self, iv, jv):	
		zi = self.db['Z_matrix'].shape[0]
		zj = self.db['Z_matrix'].shape[1]

		loop_count = 0
		stay_in_loop = True

		while stay_in_loop:
			result_w = minimize(self.Lagrange_W, self.db['W_matrix'], method='nelder-mead', args=(iv, jv), options={'xtol': 1e-6, 'disp': False})
			self.db['W_matrix'] = result_w.x.reshape(self.db['W_matrix'].shape)
	
			result_z = minimize(self.Lagrange_Z, self.db['Z_matrix'], method='nelder-mead', options={'xtol': 1e-6, 'disp': False})
			self.db['Z_matrix'] = result_z.x.reshape(self.db['Z_matrix'].shape)

			Z = self.db['Z_matrix']
			W = self.db['W_matrix']

			A = np.append(Z.T, np.eye( zi ), axis=0)
			B = np.append(np.zeros(Z.T.shape), np.eye( zi ), axis=0)
			C = np.append(np.eye(zj), np.zeros(Z.shape), axis=0)
			self.db['L'] = self.db['L'] + (A.dot(W)-B.dot(Z)-C)

			db['L1'] = db['L'][0:zj,:]
			db['L2'] = db['L'][zj:,:].T

			loop_count += 1
			if np.abs(np.sum(W.T.dot(Z) - np.eye(zj))) < 0.001: 
				print('Exit base on threshold')
				stay_in_loop = False
			if loop_count > 200: 
				print('Exit base on loop_count')
				stay_in_loop = False

		self.W_result = result_w
		return result_w.x.reshape(self.db['W_matrix'].shape)





if __name__ == "__main__":
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]])
	db['W_matrix'] = np.array([[1,0],[0,1],[0,0]])
	
	db['L1'] = np.array([[1,0], [0,2]])
	db['L2'] = np.array([[2,3,1],[0,0,1]])
	db['L'] = np.append(db['L1'], db['L2'].T, axis=0)
	
	iv = np.array([0])
	jv = np.array([1,2])
	esolver = exponential_solver(db, None)
	esolver.gamma_array = np.array([[0,1,2,-1]])
	esolver.run(iv,jv)
	
	esolver.Lagrange_W(db['W_matrix'], iv, jv)
	print esolver.optimal_val
	import pdb; pdb.set_trace()


