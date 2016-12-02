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

true_labels = genfromtxt('data_sets/breast-cancer-labels.csv', delimiter=',')
data = genfromtxt('data_sets/breast-cancer.csv', delimiter=',')

ASC = alt_spectral_clust(data)
db = ASC.db

ASC.set_values('q',2)
ASC.set_values('C_num',2)
ASC.set_values('sigma',6)
ASC.set_values('kernel_type','Gaussian Kernel')
ASC.run()
a = db['allocation']

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

start_time = time.time() 
ASC.run()
b = db['allocation']
#print "NMI : " , normalized_mutual_info_score(b,true_labels)
print("--- %s seconds ---" % (time.time() - start_time))


#Y = db['data'].dot(db['W_matrix'])
#group1 = Y[b == 1]
#group2 = Y[b == 2]
#plt.figure(2)
#plt.plot(group1[:,0], group1[:,1], 'bo')
#plt.plot(group2[:,0], group2[:,1], 'ro')
#plt.title('Reduced dimension to 2 alternative clustering result')
#plt.show()




##print "NMI : " , normalized_mutual_info_score(a,b)
##g_truth = np.concatenate((np.ones(100), np.zeros(100),np.ones(100), np.zeros(100)))
##a_truth = np.concatenate((np.ones(200), np.zeros(200)))
##print "NMI Against Ground Truth : " , normalized_mutual_info_score(b,a_truth)
##print db['Y_matrix']



#X = db['data']
#plt.subplot(221)
#plt.plot(X[:,1], X[:,2], 'ro')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Next most dominant relationship')
#
#plt.subplot(222)
#plt.plot(X[:,3], X[:,4], 'ro')
#plt.xlabel('Feature 3')
#plt.ylabel('Feature 4')
#plt.title('Non-dominant relationship')
#
#plt.subplot(223)
#plt.plot(X[:,5], X[:,7], 'ro')
#plt.xlabel('Feature 5')
#plt.ylabel('Feature 7')
#plt.title('Non-dominant relationship')
#
#
#plt.subplot(224)
#plt.plot(X[:,6], X[:,7], 'ro')
#plt.xlabel('Feature 6')
#plt.ylabel('Feature 7')
#plt.title('Non-dominant relationship')
#
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#plt.show()



#
#plt.figure(1)
#
#plt.subplot(311)
#plt.plot(X[:,0], X[:,1], 'bo')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('data_4.csv original plot')
#
##plt.figure(2)
#plt.subplot(312)
#group1 = X[a == 1]
#group2 = X[a == 2]
#plt.plot(group1[:,0], group1[:,1], 'bo')
#plt.plot(group2[:,0], group2[:,1], 'ro')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Original Clustering by FKDAC')
#
#
#plt.subplot(313)
#group1 = X[b == 1]
#group2 = X[b == 2]
#plt.plot(group1[:,0], group1[:,1], 'bo')
#plt.plot(group2[:,0], group2[:,1], 'ro')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Alternative Clustering by FKDAC')
#
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#plt.show()
#
import pdb; pdb.set_trace()
