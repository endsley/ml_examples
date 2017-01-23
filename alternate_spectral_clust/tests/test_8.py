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
import matplotlib 
colors = matplotlib.colors.cnames

#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/data_4.csv', delimiter=',')
Y_original = genfromtxt('data_sets/data_4_Y_original.csv', delimiter=',')
U_original = genfromtxt('data_sets/data_4_U_original.csv', delimiter=',')



ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

if True: #	Calculating the original clustering
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',0.5)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	a = db['allocation']
	
	#np.savetxt('data_4_U_original.csv', db['U_matrix'], delimiter=',', fmt='%d')

else: #	Predefining the original clustering, the following are the required settings
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',1)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.set_values('W_matrix',np.identity(db['d']))

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
print("--- %s seconds ---" % (time.time() - start_time))

#print "NMI : " , normalized_mutual_info_score(a,b)
#g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
#a_truth = np.concatenate((np.ones(200), np.zeros(200)))
#print "NMI Against Ground Truth : " , normalized_mutual_info_score(b,a_truth)
#print db['Y_matrix']


if True:	#	plot the clustering result
	X = db['data']
	plt.figure(1)
	
	plt.subplot(311)
	plt.plot(X[:,0], X[:,1], 'bo')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('data_4.csv original plot')
	
	#plt.figure(2)
	plt.subplot(312)
	idx = np.unique(a)
	for mm in idx:
		subgroup = X[a == mm]
		plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Original Clustering by FKDAC')
	
	
	plt.subplot(313)
	idx = np.unique(b)
	for mm in idx:
		subgroup = X[b == mm]
		plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Alternative Clustering by FKDAC')
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.show()



if False:	# save or load db to and from a pickle file
	plot_info = {}
	plot_info['debug_costVal'] = db['debug_costVal']
	plot_info['debug_gradient'] = db['debug_gradient']
	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )


	#db= pickle.load( open( "tmp_db.pk", "rb" ) )



if True:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('data_4.csv',fontsize=24)
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



import pdb; pdb.set_trace()
