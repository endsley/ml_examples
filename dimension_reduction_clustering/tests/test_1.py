#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
import sklearn.metrics
from drc import *
import matplotlib.pyplot as plt


#	Data generation
#	2 Gaussian distributions with uniform distribution as noise
noise_level =10 
noise_shift = (5 - noise_level)/2
n = 40
dim_feature = 2
dim_noise = 2
d = dim_feature + dim_noise

cluster_1 = 0.2*np.random.randn(n/2, dim_feature)
cluster_2 = 0.2*np.random.randn(n/2, dim_feature) + 5
noise = noise_level*np.random.rand(n, dim_noise) - noise_shift

data = np.vstack((cluster_1, cluster_2))
X = np.hstack((data, noise))

#	-----------------------------------------
k = 2


clf = KMeans(n_clusters=k)
allocation = clf.fit_predict(X)
print 'Pure K means :\n\t' , allocation
plt.subplot(221)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('K means')



clf = SpectralClustering(n_clusters=k, gamma=0.5)
allocation = clf.fit_predict(X)
print 'Pure Spectral Clustering :\n\t' ,allocation
plt.subplot(222)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('Spectral Clustering')



result = drc(X, 2, 0.5)
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

