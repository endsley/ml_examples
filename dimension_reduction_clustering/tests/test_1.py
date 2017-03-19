#!/usr/bin/python

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
import sklearn.metrics
import pickle

from sklearn.mixture import GMM
from drc import *
import matplotlib.pyplot as plt


##	Data generation
##	2 Gaussian distributions with uniform distribution as noise
#noise_level =12 
#noise_shift = (5 - noise_level)/2
n = 60
#dim_feature = 2
#dim_noise = 2
#d = dim_feature + dim_noise
#
#cluster_1 = 0.2*np.random.randn(n/2, dim_feature)
#cluster_2 = 0.2*np.random.randn(n/2, dim_feature) + 5
#noise = noise_level*np.random.rand(n, dim_noise) - noise_shift
#data = np.vstack((cluster_1, cluster_2))
#X = np.hstack((data, noise))



labels = np.hstack((np.zeros(n/2), np.ones(n/2)))
#pickle.dump( X , open( "./dataset/dat_1.p", "wb" ) )
X = pickle.load( open( "./dataset/dat_1.p", "rb" ) )
#	-----------------------------------------
k = 2

gmm = GMM(n_components=k)
gmm.fit(X)
allocation = gmm.predict(X)
gmm_nmi = normalized_mutual_info_score(allocation, labels)

plt.suptitle('2 Gaussians with uniform noise, First 2 dimension data 2nd 2 dimensions are noise')
plt.subplot(221)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('GMM / nmi against label : ' + str(gmm_nmi))



clf = KMeans(n_clusters=k)
allocation = clf.fit_predict(X)
kmeans_nmi = normalized_mutual_info_score(allocation, labels)

plt.subplot(222)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('Kmeans / nmi against label : ' + str(kmeans_nmi))



clf = SpectralClustering(n_clusters=k, gamma=0.5)
allocation = clf.fit_predict(X)
spectral_nmi = normalized_mutual_info_score(allocation, labels)
print 'Pure Spectral Clustering :\n\t' ,allocation
plt.subplot(223)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('Spectral Clustering \n nmi against label : ' + str(spectral_nmi))



result = drc(X, 2, 0.5)
allocation = result['allocation']
drc_nmi = normalized_mutual_info_score(allocation, labels)
print 'DRC :\n\t' , result['allocation']
print 'Dimension :\n' , result['L']
plt.subplot(224)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='o', linestyle='None')
subgroup2 = X[allocation == 1]
plt.text(3, 1, str(result['L']), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='o', linestyle='None')
plt.title('DRC / nmi against label : ' + str(drc_nmi))

plt.show()

