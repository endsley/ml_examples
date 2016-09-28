#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
import csv
from StringIO import StringIO


class random_min_finder:
	def __init__(self, db, iv, jv):
		self.db = db
		self.iv = iv
		self.jv = jv

		self.optimal_val = 0
		self.current_cost = 0
		self.gamma_array = 0

		self.W_shape = db['W_matrix'].shape
		self.wi = self.W_shape[0]
		self.wj = self.W_shape[1]


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

	def Lagrange_W(self, W):
		Wsh = self.W_shape
		W2 = W.reshape(Wsh)
		db = self.db	

		#	Setting up the cost function
		cost_foo = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.create_gamma_ij(db, 0, i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

	
		return cost_foo

	def gen_random_W(self):
		random_matrix = np.random.normal(0,10, (self.db['d'], self.db['q']) )
		q, r = np.linalg.qr(random_matrix)
		return q


	def run(self):	
		for m in range(300000):
			W = self.gen_random_W()
			cost = self.Lagrange_W(W)
			if cost < self.optimal_val:
				self.optimal_val = cost
				self.db['W_matrix'] = W
				print cost
		
		return self.optimal_val





def test_1():
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]])
	db['W_matrix'] = np.array([[1,0],[0,1],[0,0]])
	
	iv = np.array([0])
	jv = np.array([1,2])
	rmf = random_min_finder(db, iv, jv)
	
	#W = np.array([[ 0.57643199, -0.14371058], [-0.57616693,  0.62657744], [-0.57944614, -0.76599473]], dtype='f')
	#cost = rmf.Lagrange_W(W)
	print rmf.run()
	print db['W_matrix']
	import pdb; pdb.set_trace()
	
	
def test_2():
	q = 4		# the dimension you want to lower it to

	fin = open('data_1.csv','r')
	data = fin.read()
	fin.close()

	db = {}
	db['data'] = np.genfromtxt(StringIO(data), delimiter=",")
	db['N'] = db['data'].shape[0]
	db['d'] = db['data'].shape[1]
	db['q'] = q


	db['Z_matrix'] = np.identity(db['d'])[:,0:db['q']]
	db['W_matrix'] = db['Z_matrix']
	
	db['SGD_size'] = db['N']
	db['sigma'] = np.sqrt(1/2.0)
	db['maximum_W_update_count'] = 100
	
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])


	rmf = random_min_finder(db, iv, jv)
	rmf.gamma_array = np.array([[0,1,2,-1,-1,-2], [3,1,-3,-4,0,2], [1,2,3,8,5,1], [1,2,3,8,5,1], [1,0,0,8,0,0], [-1,-2,2,1,-5,0]])
	print rmf.run()
	print db['W_matrix']
	import pdb; pdb.set_trace()


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)

test_2()
