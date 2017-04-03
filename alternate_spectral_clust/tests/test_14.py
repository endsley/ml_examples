#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *
import matplotlib.pyplot as plt



university_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_univ.csv', delimiter=',')
topic_labels  = genfromtxt('data_sets/webkb_processed/webkbRaw_label_topic.csv', delimiter=',')
data = genfromtxt('data_sets/webkb_processed/min_words.csv', delimiter=',')

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma_value = np.median(d_matrix)

ASC = alt_spectral_clust(data)
db = ASC.db

ASC.set_values('q',4)
ASC.set_values('C_num',4)
ASC.set_values('sigma',sigma_value)
ASC.set_values('kernel_type','Gaussian Kernel')
start_time = time.time() 
ASC.run()
a = db['allocation']
print "University NMI : " , normalized_mutual_info_score(a,university_labels)
print "topic NMI : " , normalized_mutual_info_score(a, topic_labels)
print("Time for spectral Clustering : %s seconds ---" % (time.time() - start_time))



#orig_W = db['W_matrix']
#print 'Original allocation :' , a
#Y = db['data'].dot(orig_W)
#group1 = Y[a == 1]
#group2 = Y[a == 2]
#plt.figure(1)
#plt.plot(group1[:,0], group1[:,1], 'bo')
#plt.plot(group2[:,0], group2[:,1], 'ro')
#plt.title('Reduced dimension to 2 original clustering result, NMI=0.82')
#plt.show()
#print "NMI : " , normalized_mutual_info_score(a,true_labels)

###b = np.concatenate((np.zeros(200), np.ones(200)))
###print "NMI : " , normalized_mutual_info_score(a,b)
#

#start_time = time.time() 
#ASC.run()
#b = db['allocation']
#cf = db['cf']
#print("--- %s seconds ---" % (time.time() - start_time))
#print "alt vs truth NMI : " , normalized_mutual_info_score(b,true_labels)
#print "original vs alt : " , normalized_mutual_info_score(a,b)
#print 'My cost : ' , cf.calc_cost_function(db['W_matrix'], Y_columns=db['C_num'])
#
#
##Y = db['data'].dot(db['W_matrix'])
##group1 = Y[b == 1]
##group2 = Y[b == 2]
##plt.figure(2)
##plt.plot(group1[:,0], group1[:,1], 'bo')
##plt.plot(group2[:,0], group2[:,1], 'ro')
##plt.title('Reduced dimension to 2 alternative clustering result')
##plt.show()
#
#
#
#
###print "NMI : " , normalized_mutual_info_score(a,b)
###g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
###a_truth = np.concatenate((np.ones(200), np.zeros(200)))
###print "NMI Against Ground Truth : " , normalized_mutual_info_score(b,a_truth)
###print db['Y_matrix']
#
#
#
##X = db['data']
##plt.subplot(221)
##plt.plot(X[:,1], X[:,2], 'ro')
##plt.xlabel('Feature 1')
##plt.ylabel('Feature 2')
##plt.title('Next most dominant relationship')
##
##plt.subplot(222)
##plt.plot(X[:,3], X[:,4], 'ro')
##plt.xlabel('Feature 3')
##plt.ylabel('Feature 4')
##plt.title('Non-dominant relationship')
##
##plt.subplot(223)
##plt.plot(X[:,5], X[:,7], 'ro')
##plt.xlabel('Feature 5')
##plt.ylabel('Feature 7')
##plt.title('Non-dominant relationship')
##
##
##plt.subplot(224)
##plt.plot(X[:,6], X[:,7], 'ro')
##plt.xlabel('Feature 6')
##plt.ylabel('Feature 7')
##plt.title('Non-dominant relationship')
##
##plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
##plt.show()
#
#
#if True:	# save or load db to and from a pickle file
#	plot_info = {}
#	plot_info['debug_costVal'] = db['debug_costVal']
#	plot_info['debug_gradient'] = db['debug_gradient']
#	plot_info['debug_debug_Wchange'] = db['debug_debug_Wchange']
#	pickle.dump( plot_info, open( "tmp_db.pk", "wb" ) )
#
#	#db= pickle.load( open( "tmp_db.pk", "rb" ) )
#
#
#if True:	# plot the W convergence results
#	X = db['data']
#	plt.figure(2)
#	
#	plt.suptitle('Breast Cancer data',fontsize=24)
#	plt.subplot(311)
#	inc = 0
#	for costs in db['debug_costVal']: 
#		xAxis = np.array(range(len(costs))) + inc; 
#		inc = np.amax(xAxis)
#		plt.plot(xAxis, costs, 'b')
#		plt.plot(inc, costs[-1], 'bo', markersize=10)
#		plt.title('Cost vs w iteration, each dot is U update')
#		plt.xlabel('w iteration')
#		plt.ylabel('cost')
#		
#	plt.subplot(312)
#	inc = 0
#	for gradient in db['debug_gradient']: 
#		xAxis = np.array(range(len(gradient))) + inc; 
#		inc = np.amax(xAxis)
#		plt.plot(xAxis, gradient, 'b')
#		plt.plot(inc, gradient[-1], 'bo', markersize=10)
#		plt.title('Gradient vs w iteration, each dot is U update')
#		plt.xlabel('w iteration')
#		plt.ylabel('gradient')
#
#
#	plt.subplot(313)
#	inc = 0
#	for wchange in db['debug_debug_Wchange']: 
#		xAxis = np.array(range(len(wchange))) + inc; 
#		inc = np.amax(xAxis)
#		plt.plot(xAxis, wchange, 'b')
#		plt.plot(inc, wchange[-1], 'bo', markersize=10)
#		plt.title('|w_old - w_new|/|w| vs w iteration, each dot is U update')
#		plt.xlabel('w iteration')
#		plt.ylabel('|w_old - w_new|/|w| ')
#
#	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#	plt.subplots_adjust(top=0.85)
#	plt.show()
#
#
#
#
#
import pdb; pdb.set_trace()
