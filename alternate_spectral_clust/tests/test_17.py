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
import calc_cost
from HSIC import *
from sklearn import preprocessing


fsize = '100x4'
file_name = 'moon_' + fsize
data = genfromtxt('data_sets/' + file_name + '.csv', delimiter=',')		
data = preprocessing.scale(data)
#data[:,6] = data[:,6]/3.0
#data[:,5] = data[:,5]/3.0
#data[:,4] = data[:,4]/3.0

label_1 = genfromtxt('data_sets/moon_' + fsize + '_original_label.csv', delimiter=',')		
label_2 = genfromtxt('data_sets/moon_' + fsize + '_alt_label.csv', delimiter=',')		

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma = np.median(d_matrix)
l = 1

#	Constructor
ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db


if True:	#	initial clustering
	#	Run the original clustering
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



if True: #	Run the alternative clustering
	ASC.set_values('q',2)
	ASC.set_values('lambda', l)
	ASC.set_values('sigma',0.1)
	db['W_opt_technique'] = 'ISM'  # DG, SM, or ISM
	db['Experiment_name'] = file_name  

	#	only matters if picking DG
	#db['DG_init_W_from_pickle'] = True
	#db['pickle_count'] = 9
	
	
	start_time = time.time() 
	ASC.run()
	db['run_alternative_time'] = (time.time() - start_time)
	print("--- %s seconds ---" % db['run_alternative_time'])
	b = db['allocation']


#np.savetxt('Moon_label_2.csv', db['Y_matrix'][:,2:4], delimiter=',', fmt='%d')

#	Output the result to a file
if True:
	cf = db['cf']
	final_cost = cf.calc_cost_function(db['W_matrix'])
	against_truth = np.round(normalized_mutual_info_score(label_2, b),3)
	against_alternative = np.round(normalized_mutual_info_score(label_1, b),3)

	outLine = str(db['N']) + '\t' + db['W_opt_technique'] + '\t' + str(np.round(db['run_alternative_time'],3)) + '\t'
	outLine += str(np.round(final_cost ,3)) + '\t' + str(np.round(cf.cluster_quality(db), 3)) + '\t'
	outLine += str(against_truth) + '\t' + str(against_alternative) + '\n' 

	fin = open('moon_' + str(db['N']) + 'x' + str(db['d']) + '_result.txt','a')
	fin.write(outLine)
	fin.close()



if False:	# some HSIC debug stuff
	original_allocation = label_1
	alternate_allocation = label_2

	cf = db['cf']
	print "Original NMI : " , normalized_mutual_info_score(a,original_allocation)
	print "Alternate NMI : " , normalized_mutual_info_score(b,alternate_allocation)
	print "Alternate against original : " , normalized_mutual_info_score(b,a)

	print 'My cost : ' , cf.calc_cost_function(db['W_matrix'])
	print 'test cost : ' , calc_cost.calc_cost_function(db)

	print 'Clustering Quality : ' , cf.cluster_quality(db)
	print 'Alternative Quality : ' , l*cf.alternative_quality(db)

	print db['W_matrix']

if True:	# plot the clustering results
	X = db['data']
	plt.figure(1)
	
#	plt.subplot(221)
#	plt.plot(X[:,0], X[:,1], 'bo')
#	#plt.plot(X[:,2], X[:,3], 'bo')
#	plt.xlabel('Feature 1')
#	plt.ylabel('Feature 2')
#	plt.title('Moon dataset')
#	
#	plt.subplot(222)
#	plt.plot(X[:,2], X[:,3], 'bo')
#	#plt.plot(X[:,0], X[:,1], 'bo')
#	plt.xlabel('Feature 3')
#	plt.ylabel('Feature 4')
#	plt.title('Moon dataset')
	
	
	#plt.figure(2)
	plt.subplot(211)
	Uq_a = np.unique(a)
	group1 = X[a == Uq_a[0]]
	group2 = X[a == Uq_a[1]]

	#plt.plot(group1[:,0], group1[:,1], 'bo')
	#plt.plot(group2[:,0], group2[:,1], 'ro')
	plt.plot(group1[:,2], group1[:,3], 'bo')
	plt.plot(group2[:,2], group2[:,3], 'ro')

	#plt.xlabel('Feature 3')
	#plt.ylabel('Feature 4')
	plt.title('Original Clustering')
	
	
	plt.subplot(212)
	Uq_b = np.unique(b)
	group1 = X[b == Uq_b[0]]
	group2 = X[b == Uq_b[1]]
	#plt.plot(group1[:,2], group1[:,3], 'bo')
	#plt.plot(group2[:,2], group2[:,3], 'ro')
	plt.plot(group1[:,0], group1[:,1], 'bo')
	plt.plot(group2[:,0], group2[:,1], 'ro')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Alternative Clustering')
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.show()



if False:	# plot the W convergence results
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
