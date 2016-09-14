#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize


class random_min_finder:
	def __init__(self, db, iv, jv):
		self.db = db
		self.iv = iv
		self.jv = jv

		self.optimal_val = 0
		self.current_cost = 0

		self.W_shape = db['W_matrix'].shape
		self.wi = self.W_shape[0]
		self.wj = self.W_shape[1]

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
		Wsh = self.W_shape

		W2 = W.reshape(Wsh)
		one_matrix = np.ones((self.wj, self.wj))
		eye_matrix = np.eye(self.wj)
	
		#	Setting up the cost function
		cost_foo = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

	
		return cost_foo

	def gen_random_W(self):
		#random_matrix = np.random.rand(3,2)
		random_matrix = 10*np.random.normal(0,1, (3,2) )

		q, r = np.linalg.qr(random_matrix)
		return q

	def run(self):	

		for m in range(500000):
			W = self.gen_random_W()
			cost = self.Lagrange_W(W)
			if cost < self.optimal_val:
				self.optimal_val = cost
				print cost
	
		return self.optimal_val






db = {}
db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]])
db['W_matrix'] = np.array([[1,0],[0,1],[0,0]])

db['L1'] = np.array([[1,0], [0,2]])
db['L2'] = np.array([[2,3,1],[0,0,1]])
db['L'] = np.append(db['L1'], db['L2'].T, axis=0)


iv = np.array([0])
jv = np.array([1,2])
rmf = random_min_finder(db, iv, jv)

rmf.run()

import pdb; pdb.set_trace()


