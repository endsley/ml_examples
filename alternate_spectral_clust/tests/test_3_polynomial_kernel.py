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

X = genfromtxt('data_sets/data_1.csv', delimiter=',')

ASC = alt_spectral_clust(X)
ASC.run()
db = ASC.db

print 'U matrix\n'
print db['U_matrix'], '\n'

print 'allocation\n'
print db['allocation'], '\n'

print 'Y_matrix\n'
print db['Y_matrix'], '\n'



#	Just in case, I made some changes, the answer should still be the same as what's pickled

#####pickle.dump( db, open( "data_sets/test_3_results.pickle", "wb" ) )
#db = pickle.load( open( "data_sets/test_3_results.pickle", "rb" ) )
#print 'U matrix\n'
#print db['U_matrix'], '\n'

#print 'allocation\n'
#print db['allocation'], '\n'
#
#print 'Y_matrix\n'
#print db['Y_matrix'], '\n'


