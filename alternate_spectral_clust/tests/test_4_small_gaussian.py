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
data = genfromtxt('data_sets/data_2.csv', delimiter=',')
ASC = alt_spectral_clust(data)
ASC.set_values('q',2)
ASC.run()

db = ASC.db
print db['Y_matrix']
import pdb; pdb.set_trace()
