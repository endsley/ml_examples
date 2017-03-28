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


#data = genfromtxt('data_sets/moon.csv', delimiter=',')		
data = genfromtxt('data_sets/moon_164x7.csv', delimiter=',')		
label_1 = genfromtxt('data_sets/Moon_label_1.csv', delimiter=',')		
label_2 = genfromtxt('data_sets/Moon_label_2.csv', delimiter=',')		



#	Constructor
ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db


if True:	#	initial clustering
	#	Run the original clustering
	ASC.set_values('q',2)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',2)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	a = db['allocation']
	#np.savetxt('Moon_label_1.csv', db['Y_matrix'], delimiter=',', fmt='%d')

else: #	Predefining the original clustering, the following are the required settings
	print ':::::   USE PRE-DEFINED CLUSTERING :::::::::\n\n'
	ASC.set_values('q',2)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',2)
	ASC.set_values('lambda',1)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.set_values('W_matrix',np.identity(db['d']))

	db['Y_matrix'] = label_1
	db['U_matrix'] = label_1
	db['prev_clust'] = 1
	db['allocation'] = Y_2_allocation(label_1)
	a = db['allocation']
	#print 'Predefined allocation :' , a , '\n'



#	Run the alternative clustering
ASC.set_values('sigma',0.3)
start_time = time.time() 
ASC.run()
db['run_alternative_time'] = (time.time() - start_time)
print("--- %s seconds ---" % db['run_alternative_time'])
b = db['allocation']

#np.savetxt('Moon_label_2.csv', db['Y_matrix'][:,2:4], delimiter=',', fmt='%d')


if True:	# plot the clustering results
	X = db['data']
	plt.figure(1)
	
	plt.subplot(221)
	plt.plot(X[:,0], X[:,1], 'bo')
	#plt.plot(X[:,2], X[:,3], 'bo')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Moon dataset')
	
	plt.subplot(222)
	plt.plot(X[:,2], X[:,3], 'bo')
	#plt.plot(X[:,0], X[:,1], 'bo')
	plt.xlabel('Feature 3')
	plt.ylabel('Feature 4')
	plt.title('Moon dataset')
	
	
	#plt.figure(2)
	plt.subplot(224)
	Uq_a = np.unique(a)
	group1 = X[a == Uq_a[0]]
	group2 = X[a == Uq_a[1]]

	#plt.plot(group1[:,0], group1[:,1], 'bo')
	#plt.plot(group2[:,0], group2[:,1], 'ro')
	plt.plot(group1[:,2], group1[:,3], 'bo')
	plt.plot(group2[:,2], group2[:,3], 'ro')

	plt.xlabel('Feature 3')
	plt.ylabel('Feature 4')
	plt.title('Original Clustering by FKDAC')
	
	
	plt.subplot(223)
	Uq_b = np.unique(b)
	group1 = X[b == Uq_b[0]]
	group2 = X[b == Uq_b[1]]
	#plt.plot(group1[:,2], group1[:,3], 'bo')
	#plt.plot(group2[:,2], group2[:,3], 'ro')
	plt.plot(group1[:,0], group1[:,1], 'bo')
	plt.plot(group2[:,0], group2[:,1], 'ro')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Alternative Clustering by FKDAC')
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.show()



if True:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('The moon experiment',fontsize=24)
	plt.subplot(311)
	inc = 0
	for costs in db['debug_costVal']: 
		xAxis = np.array(range(len(costs))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, costs, 'b')
		plt.plot(inc, costs[-1], 'bo', markersize=10)
		plt.title('Cost vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('cost')
		
	plt.subplot(312)
	inc = 0
	for gradient in db['debug_gradient']: 
		xAxis = np.array(range(len(gradient))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, gradient, 'b')
		plt.plot(inc, gradient[-1], 'bo', markersize=10)
		plt.title('Gradient vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('gradient')


	plt.subplot(313)
	inc = 0
	for wchange in db['debug_debug_Wchange']: 
		xAxis = np.array(range(len(wchange))) + inc; 
		inc = np.amax(xAxis)
		plt.plot(xAxis, wchange, 'b')
		plt.plot(inc, wchange[-1], 'bo', markersize=10)
		plt.title('|w_old - w_new|/|w| vs w iteration, each dot is U update')
		plt.xlabel('w iteration')
		plt.ylabel('|w_old - w_new|/|w| ')

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.subplots_adjust(top=0.85)
	plt.show()



	#self.db['debug_costVal'] = []
	#self.db['debug_gradient'] = []
	#self.db['debug_debug_Wchange'] = []


import pdb; pdb.set_trace()
