#!/usr/bin/python

import numpy as np
import pdb
import csv
from create_y_tilde import *
from create_gamma_ij import *
from StringIO import StringIO
from sklearn.preprocessing import normalize


class direct_GD:
	def __init__(self, db, iv, jv):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]
		self.q = db['W_matrix'].shape[1]
		self.iv = iv
		self.jv = jv
		self.sigma2 = np.power(db['sigma'],2)
		self.gamma_array = None

		self.y_tilde = None
		self.W = None
		self.A = np.zeros((self.N, self.N, self.d, self.d))
		self.gamma = np.zeros((self.N, self.N))

	def create_gamma_ij(self, db, i, j):
		if type(self.gamma_array) == type(None):
			return create_gamma_ij(db, self.y_tilde, i, j)
		else:
			return self.gamma_array[i,j]

	def calc_cost_function(self, W):
		#	Calculate dL/dw gradient
		db = self.db
		iv_all = np.array(range(self.N))
		jv_all = iv_all

		#	Calc Base
		cost = 0
		for i in self.iv:
			for j in self.jv:
				
				x_dif = db['data'][i] - db['data'][j]
				x_dif = x_dif[np.newaxis]
			
				gamma_ij = self.gamma[i][j]
				cost = cost -  gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

		return cost

	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def create_A(self):
		for i in self.iv:
			for j in self.jv:
				self.A[i][j] = self.create_A_ij_matrix(i,j)

	def create_gamma(self):
		for i in self.iv:
			for j in self.jv:
				self.gamma[i][j] = self.create_gamma_ij(self.db, i,j)

	def calc_gradient(self, W):
		matrix_sum = np.zeros((self.d, self.d))
		for i in self.iv:
			for j in self.jv:
				gamma_ij = self.gamma[i][j]
				A_ij = self.A[i][j]
				exponent_term = np.exp(-0.5*np.sum(A_ij.T*(W.dot(W.T)))/self.sigma2)
				matrix_sum += gamma_ij*exponent_term*A_ij


		new_gradient = matrix_sum.dot(W)
		return new_gradient

	def run(self):
		W = self.db['W_matrix']
		self.create_A()
		self.create_gamma()


		best_cost = self.calc_cost_function(W)
		b = 0.8


		for n in range(5):
			new_grad = self.calc_gradient(W)
			new_grad = normalize(new_grad, norm='l2', axis=0)
			import pdb; pdb.set_trace()
			for m in range(9):
				new_W = W - np.power(b,m)*new_grad
				new_W = normalize(new_W, norm='l2', axis=0)

				new_cost = self.calc_cost_function(new_W)
				if(new_cost < best_cost):
					best_cost = new_cost
					W = new_W
					break

			import pdb; pdb.set_trace()
			print 'best_cost : ' , best_cost , 'new_cost : ', new_cost

		db['W_matrix'] = W
		return db['W_matrix']

def W_optimize_Gaussian(db):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	dg = direct_GD(db, iv, jv)
	dg.y_tilde = create_y_tilde(db)
	
	db['W_matrix'] = dg.run()
	
	#print sdg.W
	#print sdg.calc_cost_function(sdg.W)
	#pdb.set_trace()

def test_1():		# optimal = 2.4309
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['W_matrix'] = np.array([[1,0],[1,1],[0,0]])
	db['sigma'] = 1
	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")

	iv = np.array([0])
	jv = np.array([1,2])
	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2]])
	print sdg.run()
	
	print sdg.calc_cost_function(sdg.W)

	pdb.set_trace()


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
		
	db['SGD_size'] = db['N']
	db['sigma'] = 1
	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")

	iv = np.arange(db['N'])
	jv = np.arange(db['N'])
	db['W_matrix'] = np.random.normal(0,10, (db['d'], db['q']) )


	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2,1,1,2], [3,1,3,4,0,2], [1,2,3,8,5,1], [1,2,3,8,5,1], [1,0,0,8,0,0], [1,2,2,1,5,0]])
	#sdg.gamma_array = 4*np.random.rand(6,6)
	sdg.run()
		

	print sdg.W
	print sdg.calc_cost_function(sdg.W)

	import pdb; pdb.set_trace()

#test_2()
