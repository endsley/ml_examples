#!/usr/bin/python

import numpy as np
import pdb


class SDG:
	def __init__(self, db, iv, jv):
		self.db = db
		self.N = db['data'].shape[0]
		self.d = db['data'].shape[1]
		self.q = db['W_matrix'].shape[1]
		self.iv = iv
		self.jv = jv
		self.gamma_array = None

	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def run(self):
		A_ij = np.zeros((self.d, self.d))
		for i in self.iv:
			for j in self.jv:
				A_ij += self.create_A_ij_matrix(i,j)
		
		[U,S,V] = np.linalg.svd(A_ij)

		pdb.set_trace()

def test_1():		# optimal = 2.4309
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	db['W_matrix'] = np.array([[1,0],[1,1],[0,0]])
		
	iv = np.array([0])
	jv = np.array([1,2])
	sdg = SDG(db, iv, jv)
	sdg.gamma_array = np.array([[0,1,2]])
	sdg.run()
	
	#print 'T cost: ' , esolver.Lagrange_W(db['W_matrix'])
	#print esolver.current_cost
	pdb.set_trace()


