
import numpy as np
import sys
sys.path.append('./lib/c_libs')
from optimize_linear_kernel import *
from optimize_gaussian_kernel import *
from normalize_each_U_row import *
from optimize_polynomial_kernel import *
from K_means import *
from numpy import genfromtxt
import Nice4Py


class alt_spectral_clust:
	def __init__(self, data_set):
		self.db = {}
		self.db['C_num'] = 2				# Number of clusters

		if type(data_set) == type({}):
			data_set = np.array(data_set)

		data_dimension = data_set.shape
		self.db['N'] = data_dimension[0]
		self.db['d'] = data_dimension[1]

		self.kdac = Nice4Py.KDAC('gpu')

		self.params = {}
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

		self.db['data'] = data_set.astype(np.float32)


		self.translation = {}
		self.translation['C_num'] = 'c'
		self.translation['sigma'] = 'sigma'
		self.translation['q'] = 'q'
		self.translation['lambda'] = 'lambda'
		self.translation['verbose'] = 'verbose'
		self.translation['kernel_type'] = 'kernel'

		print '\n\nRan inside GPU\n\n'

	def remove_previous_Y_columns(self):
		self.kdac.DiscardLastRun()



	def set_values(self, key, val):

		if key in self.translation:
			self.params[self.translation[key]] = val
			if(key == 'sigma'): self.params['kernel'] = 'Gaussian'

		self.db[key] = val


	
	def center_data(self, data_set):
		# Centering matrix = I - (1/n)[ones(n,n)]
		N = self.db['N']
		centering_matrix = np.eye(N) - (1.0/N)*np.ones((N,N))

		return np.dot(centering_matrix, data_set)

	def run(self):
		db = self.db
		N = db['N']
		
		if type(db['H_matrix']) == type(None):
			db['H_matrix'] = np.eye(N) - np.ones((N,N))/N

		if db['kernel_type'] == 'Linear Kernel':
			optimize_linear_kernel(db)
		elif db['kernel_type'] == 'Gaussian Kernel':
			output = np.empty((N, 1), dtype=np.float32)

			if self.db['prev_clust'] == 0 : 
				self.kdac.SetupParams(self.params)
				self.kdac.Fit(db['data'], N, db['d'])
			else : 
				#import pdb; pdb.set_trace()
				self.kdac.SetupParams(self.params)
				self.kdac.Fit()

			self.kdac.Predict(output, N, 1)

			db['allocation'] = output.T[0]
			db['allocation'].astype(np.int32)
			db['allocation'] += 1		# starts from 1 instead of 0
		
			db['binary_allocation'] = np.zeros( ( N, db['C_num'] ) )
		
			#	Convert from allocation to binary_allocation
			for m in range(db['allocation'].shape[0]):
				db['binary_allocation'][m, int(db['allocation'][m]) - 1 ] = 1
		
			if db['Y_matrix'].shape[0] == 0:
				db['Y_matrix'] = db['binary_allocation']
			else:
				db['Y_matrix'] = np.append( db['Y_matrix'] , db['binary_allocation'], axis=1)

			self.db['prev_clust'] += 1
			return


		elif self.db['kernel_type'] == 'Polynomial Kernel':
			optimize_polynomial_kernel(self.db)
		else :
			raise ValueError('Error : unknown kernel was used.')
	
		normalize_each_U_row( self.db )
		K_means(self.db)	
		self.db['prev_clust'] += 1

