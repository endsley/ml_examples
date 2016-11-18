#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn

import time 


#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/data_4.csv', delimiter=',')
#data = genfromtxt('data_sets/Four_gaussian_3D.csv', delimiter=',')		# -0.0858 , -0.9

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',1)
ASC.set_values('C_num',2)
ASC.set_values('sigma',1)
ASC.set_values('kernel_type','Gaussian Kernel')
ASC.run()
a = db['allocation']
#b = np.concatenate((np.zeros(200), np.ones(200)))
#print a
#print "NMI : " , normalized_mutual_info_score(a,b)

start_time = time.time() 
ASC.run()
b = db['allocation']

#print("--- %s seconds ---" % (time.time() - start_time))
#print "NMI : " , normalized_mutual_info_score(a,b)
#g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
#print "NMI Against Ground Truth : " , normalized_mutual_info_score(b,g_truth)

#print db['Y_matrix']

import pdb; pdb.set_trace()
