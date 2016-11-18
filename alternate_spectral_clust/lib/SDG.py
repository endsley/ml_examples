#!/usr/bin/python

import numpy as np
import pdb
import csv
from create_y_tilde import *
from create_gamma_ij import *
from StringIO import StringIO
from scipy.optimize import minimize


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

		self.y_tilde = None
		self.W = None

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
			
				gamma_ij = self.create_gamma_ij(db, i, j)
				cost = cost -  gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

		return cost

	def check_positive_hessian(self, w):
		Hessian = np.zeros((self.d, self.d))

		for i in self.iv:
			for j in self.jv:

				gamma_ij = self.create_gamma_ij(self.db, i, j)/self.sigma2
				A_ij = self.create_A_ij_matrix(i,j)
				exponent_term = np.exp(-0.5*w.dot(A_ij).dot(w)/self.sigma2)

				p = A_ij.dot(w)
				Hessian += (gamma_ij/self.sigma2)*exponent_term*(A_ij - (1/self.sigma2)*p.dot(p.T))
	

		#print 'Hessian'
		#print Hessian
		#print w
		#print '\n\n\n'
	
		[eU,eigV,eV] = np.linalg.svd(Hessian)
		
		if np.min(eigV) > 0:
			print 'Positive'
		else:
			print 'Negative'
			pass




	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def run(self):
		db = self.db
		exponent_term = 1
		#W = db['W_matrix']
		W = np.zeros((db['d'], db['q']) )
		new_cost = float("inf")

		for m in range(2):
			matrix_sum = np.zeros((self.d, self.d))
			for i in self.iv:
				for j in self.jv:
					gamma_ij = self.create_gamma_ij(self.db, i, j)
					A_ij = self.create_A_ij_matrix(i,j)
					exponent_term = np.exp(-0.5*np.sum(A_ij.T*(W.dot(W.T)))/self.sigma2)

					matrix_sum += gamma_ij*exponent_term*A_ij
		

			#import pdb; pdb.set_trace()
			new_gradient = np.sum(matrix_sum.dot(W))

			if(new_cost == db['lowest_cost']):
				if np.abs(new_gradient) < np.abs(db['lowest_gradient']):	# This gives us the most room for GD improvement
					db['lowest_cost'] = new_cost
					db['lowest_gradient'] = new_gradient
					db['W_matrix'] = W


			matrix_sum = matrix_sum.dot(matrix_sum)
			[U,S,V] = np.linalg.svd(matrix_sum)
			W = np.fliplr(U)[:,0:self.q]

			#for k in range(self.q):
			#	self.check_positive_hessian(W[:,k])

			new_cost = self.calc_cost_function(W)
			cost_ratio = np.abs(new_cost - db['lowest_cost'])/np.abs(new_cost)

			exit_condition = np.linalg.norm(W - db['W_matrix'])/np.linalg.norm(W)
			if(new_cost < db['lowest_cost']):
				db['lowest_cost'] = new_cost
				db['lowest_gradient'] = new_gradient
				db['W_matrix'] = W

			print 'Sum(Aw) : ' , new_gradient, 'New cost :', new_cost, 'lowest Cost :' , db['lowest_cost'], 'Exit cond :' , exit_condition , 'Cost ratio : ' , cost_ratio
			print W.T
			if exit_condition < 0.0001: break;
			#except: pass


		import pdb; pdb.set_trace()
		result_w = minimize(self.calc_cost_function, db['W_matrix'], method='BFGS', options={'disp': True})
		optimal_val = result_w.fun	

		#pdb.set_trace()
		print 'Best : '
		print 'Gradient ' , db['lowest_gradient'] 
		print 'Cost  ' , db['lowest_cost']
		print 'Opt val ' , optimal_val
		self.W = db['W_matrix']
		return db['W_matrix']


def get_cost(db, W):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)
	sdg.y_tilde = create_y_tilde(db)
	return sdg.calc_cost_function(W)


def W_optimize_Gaussian_SDG(db):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)
	sdg.y_tilde = create_y_tilde(db)
	
	db['W_matrix'] = sdg.run()
	
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
