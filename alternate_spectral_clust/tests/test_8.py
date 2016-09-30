#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
import pickle


#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/KDAC_dat.csv', delimiter=',')
ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',2)
ASC.set_values('C_num',3)
ASC.set_values('kernel_type','Gaussian Kernel')
ASC.run()

print db['Y_matrix']
ASC.run()
print db['Y_matrix']

import pdb; pdb.set_trace()
