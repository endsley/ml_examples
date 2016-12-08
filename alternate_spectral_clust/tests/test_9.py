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


data = genfromtxt('data_sets/Four_gaussian_3D.csv', delimiter=',')	

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

if True:	#	initial clustering
	ASC.set_values('q',1)
	ASC.set_values('C_num',2)
	ASC.set_values('sigma',2)
	ASC.set_values('kernel_type','Gaussian Kernel')
	ASC.run()
	a = db['allocation']
	print 'Original allocation :' , a
	b = np.concatenate((np.zeros(200), np.ones(200)))
	print "NMI : " , normalized_mutual_info_score(a,b)

if True:	#	Alterntive clustering
	start_time = time.time() 
	ASC.set_values('sigma',1)
	ASC.run()
	b = db['allocation']
	print 'Alternate allocation :', b
	print("--- %s seconds ---" % (time.time() - start_time))
	print "NMI : " , normalized_mutual_info_score(a,b)
	g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
	print "NMI : " , normalized_mutual_info_score(b,g_truth)



if True:	#	Plot clustering results
	X = db['data']
	fig = plt.figure()
	ax = fig.add_subplot(211, projection='3d')
	
	group1 = X[a == 1]
	group2 = X[a == 2]
	
	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('FKDAC Original Clustering Four_gaussian_3D.csv')
	
	ax = fig.add_subplot(212, projection='3d')
	group1 = X[b == 1]
	group2 = X[b == 2]
	
	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('FKDAC Alternative Clustering Four_gaussian_3D.csv')
	
	plt.show()



if False:	# save or load db to and from a pickle file
	plot_info = {}
	plot_info['debug_costVal'] = db['debug_costVal']
	plot_info['debug_gradient'] = db['debug_gradient']
	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )


	#db= pickle.load( open( "tmp_db.pk", "rb" ) )



if False:	# plot the W convergence results
	X = db['data']
	plt.figure(2)
	
	plt.suptitle('Four_gaussian_3D.csv',fontsize=24)
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


