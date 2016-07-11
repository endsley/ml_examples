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
#print 'Y matrix : \n' , ASC.db['Y_matrix']
#print 'U matrix : \n' , ASC.db['U_matrix']
#print 'D matrix : \n' , ASC.db['D_matrix']
#print 'K matrix : \n' , ASC.db['Kernel_matrix']
#print 'H matrix : \n' , ASC.db['H_matrix']
ASC.run()


