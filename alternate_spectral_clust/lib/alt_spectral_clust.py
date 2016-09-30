
import numpy as np
import sys
from optimize_linear_kernel import *
from optimize_gaussian_kernel import *
from normalize_each_U_row import *
from optimize_polynomial_kernel import *
from K_means import *
from numpy import genfromtxt


class alt_spectral_clust:
	def __init__(self, data_set):
		self.db = {}
		self.db['C_num'] = 2				# Number of clusters

		if type(data_set) == type({}):
			data_set = np.array(data_set)

		print data_set.shape , '\n'

		data_dimension = data_set.shape
		self.db['N'] = data_dimension[0]
		self.db['d'] = data_dimension[1]
		
		self.db['sigma'] = 1
		self.db['poly_order'] = 2
		self.db['q'] = 1
		self.db['lambda'] = 1
		self.db['alpha'] = 0.01
		self.db['SGD_size'] = 20
		self.db['polynomial_constant'] = 1

		self.db['Kernel_matrix'] = np.zeros((self.db['N'],self.db['N']))
		self.db['prev_clust'] = 0
		self.db['Y_matrix'] = np.array([])
		self.db['kernel_type'] = 'Gaussian Kernel'
		#self.db['kernel_type'] = 'Linear Kernel'
		self.db['data_type'] = 'Feature Matrix'


		#outputs from U_optimize
		self.db['D_matrix'] = np.array([])
		self.db['U_matrix'] = np.array([])	
		self.db['W_matrix'] = np.array([])

		# output from spectral clustering
		self.db['allocation'] = np.array([])
		self.db['binary_allocation'] = np.array([[0,2,0],[8,2,0]])

		self.db['H_matrix'] = None
		self.db['maximum_W_update_count'] = 300
		self.db['data'] = data_set

	def set_values(self, key, val):
		self.db[key] = val
	
	def center_data(self, data_set):
		# Centering matrix = I - (1/n)[ones(n,n)]
		N = self.db['N']
		centering_matrix = np.eye(N) - (1.0/N)*np.ones((N,N))

		#import pdb; pdb.set_trace()

		return np.dot(centering_matrix, data_set)

	def run(self):
		db = self.db
		N = self.db['N']

		if db['data_type'] == 'Feature Matrix': 
			self.db['data'] = self.center_data(self.db['data'])

		if type(db['H_matrix']) == type(None):
			db['H_matrix'] = np.eye(N) - np.ones((N,N))/N

		if self.db['kernel_type'] == 'Linear Kernel':
			optimize_linear_kernel(self.db)
		elif self.db['kernel_type'] == 'Gaussian Kernel':
			optimize_gaussian_kernel(self.db)
		elif self.db['kernel_type'] == 'Polynomial Kernel':
			optimize_polynomial_kernel(self.db)
		else :
			raise ValueError('Error : unknown kernel was used.')
	
		normalize_each_U_row( self.db )
		K_means(self.db)	
		self.db['prev_clust'] += 1

