#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize


class exponential_solver:
	def __init__(self, db, iv, jv):
		self.db = db
		self.iv = iv
		self.jv = jv

		self.optimal_val = 0
		self.current_cost = 0

	def create_gamma_ij(self, i,j):
		gamma = np.array([[0,1,2,-1]])
		return gamma[i,j]

	def create_A_ij_matrix(i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(np.transpose(x_dif), x_dif)

	def Lagrange_W(self, W):
		Z = self.db['Z_matrix']
		L1 = self.db['L1']
		L2 = self.db['L2']

		W_shape = self.db['W_matrix'].shape
		W2 = W.reshape(W_shape)
		one_matrix = np.ones((W_shape[1],W_shape[1]))
		eye_matrix = np.eye(W_shape[1])
	
		#	Setting up the cost function
		cost_foo = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

		self.current_cost = cost_foo
		Lagrange = np.trace(L1.dot(W2.T.dot(Z) - eye_matrix)) + np.sum(L2.T*(W2 - Z))
		term1 = ( Z.T.dot(W2) - eye_matrix )
		term2 =  W2 - Z 	
		Aug_lag = np.trace(one_matrix.dot(term1*term1)) + np.sum(term2*term2)
		foo = cost_foo + Lagrange + Aug_lag
	
		return foo

	def Lagrange_Z(self, Z):
		W = self.db['W_matrix']
		L1 = self.db['L1']
		L2 = self.db['L2']

		Z_shape = self.db['Z_matrix'].shape
		Z2 = Z.reshape(Z_shape)

		one_matrix = np.ones((Z_shape[1],Z_shape[1]))
		eye_matrix = np.eye(Z_shape[1])
	
		#	Setting up the cost function
		cost_foo = 0

		Lagrange = np.trace(L1.dot(W.T.dot(Z2) - eye_matrix)) + np.sum(L2.T*(W - Z2))
		term1 = ( Z2.T.dot(W) - eye_matrix )
		term2 =  W - Z2 	
		Aug_lag = np.trace(one_matrix.dot(term1*term1)) + np.sum(term2*term2)

		foo = cost_foo + Lagrange + Aug_lag
		return foo


	def run(self):	
		db = self.db

		zi = db['Z_matrix'].shape[0]
		zj = db['Z_matrix'].shape[1]

		loop_count = 0
		stay_in_loop = True

		while stay_in_loop:
			result_w = minimize(self.Lagrange_W, db['W_matrix'], method='nelder-mead', options={'xtol': 1e-6, 'disp': False})
			db['W_matrix'] = result_w.x.reshape(db['W_matrix'].shape)
	
			result_z = minimize(self.Lagrange_Z, db['Z_matrix'], method='nelder-mead', options={'xtol': 1e-6, 'disp': False})
			db['Z_matrix'] = result_z.x.reshape(db['Z_matrix'].shape)

			Z = db['Z_matrix']
			W = db['W_matrix']

			#print Z
			A = np.append(Z.T, np.eye( zi ), axis=0)
			B = np.append(np.zeros(Z.T.shape), np.eye( zi ), axis=0)
			C = np.append(np.eye(zj), np.zeros(Z.shape), axis=0)

			db['L'] = db['L'] + (A.dot(W)-B.dot(Z)-C)
			db['L1'] = db['L'][0:zj,:]
			db['L2'] = db['L'][zj:,:].T
			
			loop_count += 1
			if np.abs(np.sum(W.T.dot(Z) - np.eye(zj))) < 0.001: 
				print('Exit base on threshold')
				stay_in_loop = False
			if loop_count > 200: 
				print('Exit base on loop_count')
				stay_in_loop = False

		self.optimal_val = result_w.fun	
		return result_w






db = {}
db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1],[0,0,1]])
db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]])
db['W_matrix'] = np.array([[1,0],[0,1],[0,0]])

db['L1'] = np.array([[1,0], [0,2]])
db['L2'] = np.array([[2,3,1],[0,0,1]])
db['L'] = np.append(db['L1'], db['L2'].T, axis=0)


iv = np.array([0])
jv = np.array([1,2])
esolver = exponential_solver(db, iv, jv)
esolver.run()

esolver.Lagrange_W(db['W_matrix'])
print esolver.current_cost
import pdb; pdb.set_trace()


