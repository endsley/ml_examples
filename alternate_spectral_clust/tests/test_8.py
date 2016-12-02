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
from cost_function import *
import matplotlib.pyplot as plt
from Y_2_allocation import *

#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/data_4.csv', delimiter=',')
Y_original = genfromtxt('data_sets/data_4_Y_original.csv', delimiter=',')

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

if False: #	Calculating the original clustering
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',1)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	a = db['allocation']
else: #	Predefining the original clustering
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',1)
	ASC.set_values('kernel_type','Gaussian Kernel')
	db['Y_matrix'] = Y_original
	db['U_matrix'] = Y_original
	db['prev_clust'] = 1
	db['allocation'] = Y_2_allocation(Y_original)
	a = db['allocation']

#b = np.concatenate((np.zeros(200), np.ones(200)))
#print "NMI : " , normalized_mutual_info_score(a,b)

start_time = time.time() 
ASC.run()
b = db['allocation']
#print 'Alternate allocation :', b
#print("--- %s seconds ---" % (time.time() - start_time))

#print "NMI : " , normalized_mutual_info_score(a,b)
#g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
#a_truth = np.concatenate((np.ones(200), np.zeros(200)))
#print "NMI Against Ground Truth : " , normalized_mutual_info_score(b,a_truth)
#print db['Y_matrix']



X = db['data']
plt.figure(1)

plt.subplot(311)
plt.plot(X[:,0], X[:,1], 'bo')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('data_4.csv original plot')

#plt.figure(2)
plt.subplot(312)
group1 = X[a == 1]
group2 = X[a == 2]
plt.plot(group1[:,0], group1[:,1], 'bo')
plt.plot(group2[:,0], group2[:,1], 'ro')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Clustering by FKDAC')


plt.subplot(313)
group1 = X[b == 1]
group2 = X[b == 2]
plt.plot(group1[:,0], group1[:,1], 'bo')
plt.plot(group2[:,0], group2[:,1], 'ro')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Alternative Clustering by FKDAC')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
plt.show()

import pdb; pdb.set_trace()
