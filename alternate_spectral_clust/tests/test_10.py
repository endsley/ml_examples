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


data = genfromtxt('data_sets/moon.csv', delimiter=',')		
#data = genfromtxt('data_sets/moon_164x7.csv', delimiter=',')		

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',2)
ASC.set_values('C_num',2)
ASC.set_values('sigma',2)
ASC.set_values('kernel_type','Gaussian Kernel')

ASC.run()
a = db['allocation']


ASC.set_values('sigma',0.3)
start_time = time.time() 
ASC.run()
print("--- %s seconds ---" % (time.time() - start_time))
b = db['allocation']

X = db['data']

plt.figure(1)

plt.subplot(221)
plt.plot(X[:,0], X[:,1], 'bo')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moon dataset')

plt.subplot(222)
plt.plot(X[:,2], X[:,3], 'bo')
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('Moon dataset')


#plt.figure(2)
plt.subplot(224)
group1 = X[a == 1]
group2 = X[a == 2]
plt.plot(group1[:,2], group1[:,3], 'bo')
plt.plot(group2[:,2], group2[:,3], 'ro')
plt.xlabel('Feature 3')
plt.ylabel('Feature 4')
plt.title('Original Clustering by FKDAC')


plt.subplot(223)
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
