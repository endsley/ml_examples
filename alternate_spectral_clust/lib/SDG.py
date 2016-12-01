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


	def check_positive_hessian(self, w):
		Hessian = np.zeros((self.d, self.d))

		for i in self.iv:
			for j in self.jv:

				gamma_ij = self.create_gamma_ij(self.db, i, j)/self.sigma2
				A_ij = self.create_A_ij_matrix(i,j)
				exponent_term = np.exp(-0.5*w.dot(A_ij).dot(w)/self.sigma2)

				p = A_ij.dot(w)
				Hessian += (gamma_ij/self.sigma2)*exponent_term*(A_ij - (1/self.sigma2)*p.dot(p.T))
	
	
		[eU,eigV,eV] = np.linalg.svd(Hessian)
		
		if np.min(eigV) > 0:
			print 'Positive'
		else:
			print 'Negative'
			pass


	def run(self):
		db = self.db
		exponent_term = 1
		W = db['W_matrix']
		#W = np.zeros((db['d'], db['q']) )
		use_frank = False
		new_cost = float("inf")

		for m in range(2):
			matrix_sum = db['cf'].create_gamma_exp_A(W)

			#if(new_cost == db['lowest_cost']):
			#	if np.abs(new_gradient) < np.abs(db['lowest_gradient']):	# This gives us the most room for GD improvement
			#		db['lowest_cost'] = new_cost
			#		db['lowest_gradient'] = new_gradient
			#		db['W_matrix'] = W




			new_gradient = matrix_sum.dot(W)
			new_gradient_mag = np.sum(new_gradient)

#			if use_frank:
##		Frank Wolfe
#				W = -new_gradient/np.linalg.norm(new_gradient)
#				use_frank = not use_frank
#				print '\t\tTrace : ' , np.trace(new_gradient.T.dot(W))
#			else:
##		My way
#			use_frank = not use_frank
			#matrix_sum = matrix_sum.dot(matrix_sum)

			[U,S,V] = np.linalg.svd(matrix_sum)
			W = np.fliplr(U)[:,0:self.q]
			W_max = U[:,0:self.q]
	
			#cost_W_max = -db['cf'].calc_cost_function(W_max)
			new_cost = -db['cf'].calc_cost_function(W)

			#if cost_W_max < new_cost:
			#	print 'W max wins'
			#	W = W_max
			#	new_cost = cost_W_max
			#else:
			#	pass
			#	#print 'W max loses'



			#pdb.set_trace()
			#new_cost = -db['cf'].calc_cost_function(W)
			cost_ratio = np.abs(new_cost - db['lowest_cost'])/np.abs(new_cost)

			exit_condition = np.linalg.norm(W - db['W_matrix'])/np.linalg.norm(W)
			if(new_cost < db['lowest_cost']):
				db['lowest_U'] = db['U_matrix']
				db['lowest_cost'] = new_cost
				db['lowest_gradient'] = new_gradient_mag
				db['W_matrix'] = W
				#import pdb; pdb.set_trace()

			print 'Sum(Aw) : ' , new_gradient_mag, 'New cost :', new_cost, 'lowest Cost :' , db['lowest_cost'], 'Exit cond :' , exit_condition , 'Cost ratio : ' , cost_ratio
			if exit_condition < 0.0001: break;
			#except: pass



		print 'Best : '
		print 'Gradient ' , db['lowest_gradient'] 
		print 'Cost  ' , db['lowest_cost']
		self.W = db['W_matrix']
		return db['W_matrix']


def get_cost(db, W):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)
	sdg.y_tilde = create_y_tilde(db)

	import pdb; pdb.set_trace()
	return sdg.calc_cost_function(W)


def W_optimize_Gaussian_SDG(db):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)
	sdg.y_tilde = create_y_tilde(db)
	
	sdg.run()
	

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
