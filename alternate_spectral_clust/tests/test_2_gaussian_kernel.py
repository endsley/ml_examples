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
#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/data_1.csv', delimiter=',')
ASC = alt_spectral_clust(data)
ASC.set_values('q',2)
ASC.run()

print 'U matrix\n'
print ASC.db['U_matrix'], '\n'

print 'allocation\n'
print ASC.db['allocation'], '\n'

print 'Y_matrix\n'
print ASC.db['Y_matrix'], '\n'


#	Just in case, I made some changes, the answer should still be the same as what's pickled

#####pickle.dump( db, open( "data_sets/test_2_results.pickle", "wb" ) )
#db = pickle.load( open( "data_sets/test_2_results.pickle", "rb" ) )
#print 'U matrix\n'
#print db['U_matrix'], '\n'

#print 'allocation\n'
#print db['allocation'], '\n'
#
#print 'Y_matrix\n'
#print db['Y_matrix'], '\n'


