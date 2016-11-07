#!/usr/bin/python

import numpy as np
import pdb
import csv
from StringIO import StringIO


class SDG:
	def __init__(self, db, iv, jv):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]
		self.q = db['W_matrix'].shape[1]
		self.iv = iv
		self.jv = jv
		self.sigma2 = np.power(db['sigma'],2)
		self.gamma_array = None
		self.W = None
		self.A = np.zeros((self.N, self.N, self.d, self.d))
		self.gamma = np.zeros((self.N, self.N))
		self.y_tilde = None
		self.exponent_term = np.zeros((self.N, self.N))

	def create_gamma_ij(self, i, j):
		if type(self.gamma_array) == type(0):
			return create_gamma_ij(self.db, self.y_tilde, i, j)
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
			
				gamma_ij = self.create_gamma_ij(i, j)
				cost = cost -  gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

		return cost

	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def create_A_matrix(self):
		for i in self.iv:
			for j in self.jv:
				x_dif = self.db['data'][i] - self.db['data'][j]
				x_dif = x_dif[np.newaxis]
				self.A[i][j] = np.dot(x_dif.T, x_dif)

	def create_gamma(self):
		for i in self.iv:
			for j in self.jv:
				self.gamma[i][j] = self.create_gamma_ij(i, j)	

	def create_exponent_term(self, W):
		for i in self.iv:
			for j in self.jv:
				self.exponent_term[i,j] = np.exp(-0.5*np.sum(self.A[i][j]*(W.dot(W.T)))/self.sigma2)


	def get_new_W(self, matrix_sum):
		[U,S,V] = np.linalg.svd(matrix_sum)
		U = np.fliplr(U)

		W = np.zeros((self.d,self.q))
		column_count = 0
		for idx in range(U.shape[1]):
			w = U[:,idx]
			self.create_exponent_term(w)
			const_term = self.gamma*self.exponent_term/self.sigma2
			Hessian = np.zeros((self.d, self.d))
			for i in self.iv:
				for j in self.jv:
					p = self.A[i][j].dot(w)
					Hessian += const_term[i][j]*(self.A[i][j] - (1/self.sigma2)*p.dot(p.T))
	
			
			[eU,eigV,eV] = np.linalg.svd(Hessian)
	
			if np.min(eigV) > 0:
				print 'Positive'
				W[:,column_count] = w
				column_count += 1
			else:
				print 'Negative'

			if(column_count == self.q): break

		return W

	def run(self):
		self.create_A_matrix()
		self.create_gamma()

		W = np.zeros((self.d,self.q))
		self.create_exponent_term(W)

		for m in range(100):
			matrix_sum = np.zeros((self.d, self.d))
			self.create_exponent_term(W)
			for i in self.iv:
				for j in self.jv:
					matrix_sum += self.gamma[i][j]*self.exponent_term[i][j]*self.A[i][j] # did not include constant sigma cus it doesn't matter

			W = self.get_new_W(matrix_sum)

			try: 
				if np.linalg.norm(W - self.W)/np.linalg.norm(W) < 0.0001: 
					break;
			except: pass

			self.W = W



def test_1():		# optimal = 2.4309
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['W_matrix'] = np.array([[1,0],[1,1],[0,0]])
	db['sigma'] = 1/np.sqrt(2)
		
	iv = np.array([0])
	jv = np.array([1,2])
	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2]])
	sdg.run()
	
	print sdg.W
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
	
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])
	db['W_matrix'] = np.random.normal(0,10, (db['d'], db['q']) )


	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2,1,1,2], [3,1,3,4,0,2], [1,2,3,8,5,1], [1,2,3,8,5,1], [1,0,0,8,0,0], [1,2,2,1,5,0]])
	sdg.run()
		

	print sdg.W
	print sdg.calc_cost_function(sdg.W)

	import pdb; pdb.set_trace()

test_2()
