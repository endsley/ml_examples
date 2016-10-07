#!/usr/bin/python

import sys
sys.path.append('./lib')

#from alt_spectral_clust import *
from alt_spectral_clust_gpu import *

import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
import pickle
import time

#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/data_4.csv', delimiter=',')
#data = genfromtxt('data_sets/Four_gaussian_3D.csv', delimiter=',')
ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',1)
ASC.set_values('C_num',2)
ASC.set_values('sigma',3)
ASC.set_values('kernel_type','Gaussian Kernel')

start = time.time()
ASC.run()
ASC.run()
end = time.time()
print db['Y_matrix']
print(end - start) , ' seconds'

import pdb; pdb.set_trace()
