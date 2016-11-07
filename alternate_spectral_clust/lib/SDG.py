#!/usr/bin/python

import numpy as np
import pdb
import csv
from create_y_tilde import *
from create_gamma_ij import *
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
		self.W = np.zeros((db['data'].shape[1], db['q']))

		self.y_tilde = None

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

	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def run(self):
		exponent_term = 1
		W = np.zeros((self.d,self.q))
		best_W = np.zeros((self.d,self.q))
		cost_function = float("inf")
		gradient = float("inf")

		for m in range(15):
			matrix_sum = np.zeros((self.d, self.d))
			A_sum = np.zeros((self.d, self.d))
			for i in self.iv:
				for j in self.jv:
					gamma_ij = self.create_gamma_ij(self.db, i, j)
					A_ij = self.create_A_ij_matrix(i,j)
					exponent_term = np.exp(-0.5*np.sum(A_ij.T*(W.dot(W.T)))/self.sigma2)

					matrix_sum += gamma_ij*exponent_term*A_ij
		
					A_sum += A_ij

			matrix_sum = matrix_sum.dot(matrix_sum)
			[U,S,V] = np.linalg.svd(matrix_sum)
			W = np.fliplr(U)[:,0:self.q]

			#W = 0.5*W_new + (1-0.5)*W		# interpolation doesn't seem to work

			new_cost = self.calc_cost_function(W)
			#print cost_function, new_cost
			cost_ratio = np.abs(new_cost - cost_function)/np.abs(new_cost)
			new_gradient = np.sum(matrix_sum.dot(W))

			if(new_cost < cost_function):
				cost_function = new_cost
				best_W = W
				gradient = new_gradient
			elif(cost_ratio < 0.001):
				if new_gradient < gradient:
					best_W = W
					gradient = new_gradient


			#if(cost_ratio < 0.0001): break
			#pdb.set_trace()

			exit_condition = np.linalg.norm(W - self.W)/np.linalg.norm(W)
			print 'Sum(Aw) : ' , np.sum(matrix_sum.dot(W)), 'New cost :', new_cost, 'Cost fun :' , cost_function, 'Exit cond :' , exit_condition , 'Cost ratio : ' , cost_ratio
			if exit_condition < 0.0001: break;
			self.W = W
			#except: pass

		self.W = best_W
		#pdb.set_trace()
		print 'Best : '
		print 'Gradient ' , gradient 
		print 'Cost  ' , cost_function
		return self.W

def W_optimize_Gaussian(db):
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

