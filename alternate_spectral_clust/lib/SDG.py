#!/usr/bin/python

import numpy as np
import pdb
import time
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

		self.costVal_list = []
		self.gradient_list = []
		self.Wchange_list = []

		self.y_tilde = None
		self.W = None

		self.debug_mode = True

	def run_debug_1(self, new_gradient_mag, new_cost, lowest_cost, exit_condition):
		if self.debug_mode:
			self.costVal_list.append(new_cost)
			self.gradient_list.append(new_gradient_mag)
			self.Wchange_list.append(exit_condition)
	
			print 'Sum(Aw) : ' , new_gradient_mag, 'New cost :', new_cost, 'lowest Cost :' , lowest_cost, 'Exit cond :' , exit_condition 

	def run_debug_2(self, db, lowest_gradient, lowest_cost):
		if self.debug_mode:
			print 'Best : '
			print 'Gradient ' , lowest_gradient
			print 'Cost  ' , lowest_cost

			self.db['debug_costVal'].append(self.costVal_list)
			self.db['debug_gradient'].append(self.gradient_list)
			self.db['debug_debug_Wchange'].append(self.Wchange_list)


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

	def update_best_W(self, new_cost, new_gradient_mag, W):
		db = self.db

		if(new_cost < db['lowest_cost']):
			db['lowest_U'] = db['U_matrix']
			db['lowest_cost'] = new_cost
			db['lowest_gradient'] = new_gradient_mag
			db['W_matrix'] = W
			#import pdb; pdb.set_trace()

	def run(self):
		db = self.db
		exponent_term = 1
		W = db['W_matrix']
		#W = np.zeros((db['d'], db['q']) )
		use_frank = False
		new_cost = float("inf")


		for m in range(5): 
			[cost, matrix_sum] = db['cf'].calc_cost_function(W, also_calc_Phi=True)

			##	Testing the new Phi matrix
			#start_time = time.time() 
			#[cost, matrix_sum_1] = db['cf'].calc_cost_function(W, also_calc_Phi=True)
			#print matrix_sum_1
			#print("--- 1 : %s seconds ---" % (time.time() - start_time))

			#start_time = time.time() 
			#matrix_sum = db['cf'].create_gamma_exp_A(W)
			#print matrix_sum
			#print("--- 2 : %s seconds ---" % (time.time() - start_time))

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

			[U,S,V] = np.linalg.svd(matrix_sum)
			#print 'Smallest eigenvalue : ' , np.min(S)
			#print 'Coef :' , U.T.dot(W) , '\n'
			reverse_S = S[::-1]
			eigsValues = reverse_S[0:self.q]

			# Check if descending
			#print "Before :" , np.trace(W.T.dot(matrix_sum).dot(W))
			#W_before = W
			W = np.fliplr(U)[:,0:self.q]
			#print "After :" , np.trace(W_before.T.dot(matrix_sum).dot(W))



			#	This is the Lagrangian gradient
			#print '\n\nGrad : ' , np.linalg.norm(new_gradient - W*eigsValues) ,'\n\n'
			#import pdb; pdb.set_trace()
	
			new_cost = db['cf'].calc_cost_function(W)

			#import pdb; pdb.set_trace()
			##	Frank Wolfe
			#if use_frank:
			#	W = -new_gradient/np.linalg.norm(new_gradient)

			if False:	# also check for max
				W_max = U[:,0:self.q]
				cost_W_max = db['cf'].calc_cost_function(W_max)
				if cost_W_max < new_cost:
					print 'W max wins'
					W = W_max
					new_cost = cost_W_max
				else:
					pass
					#print 'W max loses'


			exit_condition = np.linalg.norm(W - db['W_matrix'])/np.linalg.norm(W)
			self.update_best_W(new_cost, new_gradient_mag, W)


			self.run_debug_1(new_gradient_mag, new_cost, db['lowest_cost'], exit_condition)
			if exit_condition < 0.0001: break;
			#except: pass


		#self.run_debug_2(db, db['lowest_gradient'], Lowest_cost)
		self.run_debug_2(db, db['lowest_gradient'], db['lowest_cost'])
		Lowest_cost = db['cf'].calc_cost_function(db['W_matrix'])
		print 'Lowest cost ' , Lowest_cost , '\n\n'
		self.W = db['W_matrix']
		return db['W_matrix']


def W_optimize_Gaussian_SDG(db):
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])

	sdg = SDG(db, iv, jv)
	#sdg.y_tilde = create_y_tilde(db)	
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
