#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
import pickle

#Initialize
db = {}
db['C_num'] = 3
db['N'] = 10
db['d'] = 3

db['sigma'] = 2
db['poly_order'] = 2
db['q'] = db['d']
db['lambda'] = 1
db['alpha'] = 0.5

db['data'] = genfromtxt('data_sets/data_1.csv', delimiter=',')
db['Kernel_matrix'] = np.array([])
db['prev_clust'] = 0
db['Y_matrix'] = np.array([])
db['kernel_type'] = 'LINEAR_KERNEL'

#outputs from U_optimize
db['D_matrix'] = np.array([])
db['U_matrix'] = np.array([])

#output 
db['W_matrix'] = np.array([])

# output from spectral clustering
db['allocation'] = np.array([])
db['binary_allocation'] = np.array([[0,2,0],[8,2,0]])

alt_spectral_clust(db)
db['q'] = 2
alt_spectral_clust(db)

print 'U matrix\n'
print db['U_matrix'], '\n'

print 'allocation\n'
print db['allocation'], '\n'

print 'Y_matrix\n'
print db['Y_matrix'], '\n'

print 'W_matrix\n'
print db['W_matrix'], '\n'




#	Just in case, I made some changes, the answer should still be the same as what's pickled

#####pickle.dump( db, open( "data_sets/test_1_results.pickle", "wb" ) )
#db = pickle.load( open( "data_sets/test_1_results.pickle", "rb" ) )
#print 'U matrix\n'
#print db['U_matrix'], '\n'

#print 'allocation\n'
#print db['allocation'], '\n'
#
#print 'Y_matrix\n'
#print db['Y_matrix'], '\n'


