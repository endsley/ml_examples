#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
import sklearn.metrics
from drc import *
import matplotlib.pyplot as plt
import pickle

#	Data generation Moon
#n = 200
#x = 4*np.random.rand(n,1) - 2
#y = np.power(x,2) + 0.1*np.random.randn(n,1)
#x = x + 0.1*np.random.randn(n,1)
#dat_1 = np.hstack((x,y))
#
#a = 4*np.random.rand(n,1)
#b = -1.4*np.power((a-2),2) + 7.5 + 0.1*np.random.randn(n,1)
#a = a + 0.1*np.random.randn(n,1)
#dat_2 = np.hstack((a,b))
#
#noise = 10*np.random.rand(n*2, 3)
#
#dat = np.vstack((dat_1,dat_2))
#X = np.hstack((dat, noise))
#
#pickle.dump( X , open( "dat_2.p", "wb" ) )
#plt.plot(X[:,2], X[:,3],'x')
#plt.show()
X = pickle.load( open( "dat_2.p", "rb" ) )

#X = np.hstack((data, noise))
#
##	-----------------------------------------
#A = np.eye(d)
#H = np.eye(n) - (1.0/n)*np.ones((n,n))
#U_converged = False
#k = 2
#delta = 0.001
#
#

clf = KMeans(n_clusters=2)
allocation = clf.fit_predict(X)
print 'Pure K means :\n\t' , allocation
plt.subplot(221)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('K means')

clf = SpectralClustering(n_clusters=2, gamma=0.08)
allocation = clf.fit_predict(X)
print 'Pure Spectral Clustering :\n\t' ,allocation
plt.subplot(222)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('Spectral Clustering')


result = drc(X, 2, 0.08)
allocation = result['allocation']
print 'DRC :\n\t' , result['allocation']
print 'Dimension :\n' , result['L']

plt.subplot(223)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('DRC')

plt.show()

