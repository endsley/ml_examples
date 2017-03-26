#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn.metrics
from drc import *
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
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
X = pickle.load( open( "./dataset/small_moon.p", "rb" ) )


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


k = 2
labels = np.hstack((np.zeros(200), np.ones(200)))

gmm = GMM(n_components=k)
gmm.fit(X)
allocation = gmm.predict(X)
gmm_nmi = normalized_mutual_info_score(allocation, labels)
plt.suptitle('2 Moons with uniform noise, First 2 dimension=moon 3 dimensions after are noise')
plt.subplot(221)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='x', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='.', linestyle='None')
plt.title('GMM \n nmi against label : ' + str(gmm_nmi))




clf = KMeans(n_clusters=k)
allocation = clf.fit_predict(X)
kmeans_nmi = normalized_mutual_info_score(allocation, labels)
#print 'Pure K means :\n\t' , allocation
plt.subplot(222)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='x', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='.', linestyle='None')
plt.title('Kmeans \n nmi against label : ' + str(kmeans_nmi))


clf = SpectralClustering(n_clusters=k, gamma=0.08)
allocation = clf.fit_predict(X)
spectral_nmi = normalized_mutual_info_score(allocation, labels)
plt.subplot(223)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='x', linestyle='None')
subgroup2 = X[allocation == 1]
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='.', linestyle='None')
plt.title('Spectral Clustering \n nmi against label : ' + str(spectral_nmi))


result = drc(X, 2, 0.06)
allocation = result['allocation']
drc_nmi = normalized_mutual_info_score(allocation, labels)
#print 'DRC :\n\t' , result['allocation']
#print 'Dimension :\n' , result['L']

plt.subplot(224)
subgroup = X[allocation == 0]
plt.plot(subgroup[:,0], subgroup[:,1], color='r' , marker='x', linestyle='None')
subgroup2 = X[allocation == 1]
plt.text(2, 0, str(np.round(result['L'],2)), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
plt.plot(subgroup2[:,0], subgroup2[:,1], color='b' , marker='.', linestyle='None')
plt.title('DRC \n nmi against label : ' + str(drc_nmi))

plt.show()

