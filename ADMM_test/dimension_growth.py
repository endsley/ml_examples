#!/usr/bin/python

import numpy as np
from scipy.optimize import minimize
import csv
from StringIO import StringIO

class dimension_growth:
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
		self.y_tilde = 0

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
			
				gamma_ij = self.create_gamma_ij(self.db, 0, i, j)
				cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W2).dot(W2.T).dot(x_dif.T))

	
		return cost_foo

	def create_gamma_ij(self, db, y_tilde, i, j):
		if type(self.gamma_array) == type(0):
			return create_gamma_ij(db, self.y_tilde, i, j)
		else:
			return self.gamma_array[i,j]

	def create_A_ij_matrix(self, i, j):
		db = self.db
		x_dif = db['data'][i] - db['data'][j]
		x_dif = x_dif[np.newaxis]
		return np.dot(x_dif.T, x_dif)

	def update_previous_gw(self, w, previous_gw):
		db = self.db

		for i in range(db['N']):
			for j in range(db['N']):
				A_ij = self.create_A_ij_matrix(i,j)
	
				part_1 = np.true_divide(np.dot(A_ij, w), np.power(db['sigma'],2))
				previous_gw[i,j] = previous_gw[i,j]*np.exp(np.true_divide(np.dot(w, part_1),-2))
	
		return previous_gw

	def Stochastic_W_gradient(self, db, y_tilde, previous_gw, w_l, i_values, j_values):
		update_direction = 0
		updated_magnitude = 0

		for i in i_values:
			for j in j_values:
				A_ij = self.create_A_ij_matrix(i,j)
				gamma_ij = self.create_gamma_ij(db, self.y_tilde, i, j)
	
				part_1 = np.true_divide(np.dot(A_ij, w_l), np.power(db['sigma'],2))
				part_2 = np.exp(np.true_divide(np.dot(w_l, part_1),-2))
	
				portion_magnitude = gamma_ij*previous_gw[i,j]*part_2;
	
				ij_addition = -portion_magnitude*part_1
				updated_magnitude += portion_magnitude
				update_direction += ij_addition
		
		#print 'updated_magnitud : ' , updated_magnitude
		return [update_direction, updated_magnitude]

	def get_orthogonal_vector(self, db, m, input_vector):	
		count_down = m
	
		#	perform Gram Schmit
		while count_down != 0:
			count_down = count_down - 1
			w_prev = db['W_matrix'][:,count_down]
		
			projected_direction = (np.dot(w_prev,input_vector)/np.dot(w_prev, w_prev))*w_prev
			input_vector = input_vector - projected_direction	
		
		input_vector = input_vector/np.linalg.norm(input_vector)
		return input_vector


	def run(self):
		db = self.db
		previous_gw = np.ones((db['N'],db['N']))
		w_converged = False
		last_W = None
	
		for m in range(db['q']):
			db['W_matrix'][:,m] = self.get_orthogonal_vector(db, m, db['W_matrix'][:,m])
			counter = 0
			new_alpha = 0.5
	

			while not w_converged:
				w_l = db['W_matrix'][:,m]
	
				[update_direction, db['updated_magnitude']] = self.Stochastic_W_gradient(db, self.y_tilde, previous_gw, w_l, self.iv, self.jv)
				update_direction = self.get_orthogonal_vector(db, m+1, update_direction) # m+1 is to also remove the current dimension
	
	
				new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction	
				[tmp_dir, new_mag] = self.Stochastic_W_gradient(db, self.y_tilde, previous_gw, new_W, self.iv, self.jv)

	
				#print db['updated_magnitude'], new_mag, new_alpha
				while new_mag < db['updated_magnitude']:
					new_alpha = new_alpha * 0.8
					if new_alpha > 0.00001 :
						new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
						[tmp_dir, new_mag] = self.Stochastic_W_gradient(db, self.y_tilde, previous_gw, new_W, self.iv, self.jv)
					else:
						new_alpha = 0
						break
	
				if type(last_W) != type(None):
					relative_magnitude = np.linalg.norm(new_W)
					distance_change_since_last_update = np.linalg.norm(last_W - new_W)
	
					if (distance_change_since_last_update/relative_magnitude) < 0.001 : 
						w_converged = True
						try:
							db.pop('previous_wolfe_direction')
							db.pop('previous_wolfe_magnitude')
						except: pass
	
				#print new_W
				last_W = new_W/np.linalg.norm(new_W)
				db['W_matrix'][:,m] = last_W


				counter += 1
				if counter > db['maximum_W_update_count']: 
					print '\nExit due to maximum update reached'
					w_converged = True
	
			w_converged = False
			previous_gw = self.update_previous_gw(last_W, previous_gw)
	
		db['W_matrix'] = db['W_matrix'][:,0:db['q']]
		#print db['W_matrix']




def test_1():		#	optimal = -2.4308
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1]])
	#db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]], dtype='f' )
	db['W_matrix'] = np.array([[10,15],[10,1],[0,0]], dtype='f')
	
	db['N'] = db['data'].shape[0]
	db['SGD_size'] = db['N']
	db['q'] = db['W_matrix'].shape[1]
	db['sigma'] = np.sqrt(1/2.0)
	db['maximum_W_update_count'] = 100
	
	iv = np.array([0])
	jv = np.array([1,2])
	dgrowth = dimension_growth(db, iv, jv)
	dgrowth.gamma_array = np.array([[0,1,2,-1]])
	dgrowth.run()
	
	final_cost = dgrowth.Lagrange_W(db['W_matrix'])
	print final_cost
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
		
	db['SGD_size'] = db['N']
	db['sigma'] = np.sqrt(1/2.0)
	db['maximum_W_update_count'] = 100
	
	iv = np.arange(db['N'])
	jv = np.arange(db['N'])


	for m in range(10):
		#db['Z_matrix'] = np.random.normal(0,10, (db['d'], db['q']) )
		#db['Z_matrix'] = np.identity(db['d'])[:,0:db['q']]
		db['W_matrix'] = np.random.normal(0,10, (db['d'], db['q']) )

		dgrowth = dimension_growth(db, iv, jv)
		dgrowth.gamma_array = np.array([[0,1,2,1,1,2], [3,1,3,4,0,2], [1,2,3,8,5,1], [1,2,3,8,5,1], [1,0,0,8,0,0], [1,2,2,1,5,0]])
		dgrowth.run()
		
		final_cost = dgrowth.Lagrange_W(db['W_matrix'])
		print final_cost

	import pdb; pdb.set_trace()

def test_3():		#	optimal = -2.4308
	db = {}
	db['data'] = np.array([[3,4,0],[2,4,-1],[0,2,-1],[1,1,1]])
	db['Z_matrix'] = np.array([[1,0],[0,1],[0,0]], dtype='f' )
	db['W_matrix'] = np.array([[10,15],[10,1],[0,0]], dtype='f')
	
	db['N'] = db['data'].shape[0]
	db['SGD_size'] = db['N']
	db['q'] = db['W_matrix'].shape[1]
	db['sigma'] = np.sqrt(1/2.0)
	db['maximum_W_update_count'] = 100
	
	iv = np.array([0])
	jv = np.array([1,2,3])
	dgrowth = dimension_growth(db, iv, jv)
	dgrowth.gamma_array = np.array([[0,1,2,-1,2]])
	dgrowth.run()
	
	final_cost = dgrowth.Lagrange_W(db['W_matrix'])
	print final_cost
	import pdb; pdb.set_trace()
	
np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)

test_2()
