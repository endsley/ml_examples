#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize			# version : 0.17
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors







np.set_printoptions(precision=4)
np.set_printoptions(threshold=3000)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def spectral(X, sigma, k):
	Vgamma = 1/(2*sigma*sigma)
	return SpectralClustering(k, gamma=Vgamma).fit_predict(X)


def L_to_U(L, k):
	eigenValues,eigenVectors = np.linalg.eigh(L)

	n2 = len(eigenValues)
	n1 = n2 - k
	U = eigenVectors[:, n1:n2]
	U_lambda = eigenValues[n1:n2]
	U_normalized = normalize(U, norm='l2', axis=1)
	
	return [U, U_normalized]

def get_k_nearest_samples(X, x):
	samples = [[0, 0, 2], [1, 0, 1], [0, 0, 1], [1,1,1]]
	
	neigh = NearestNeighbors(2)
	neigh.fit(samples)
	result = neigh.kneighbors([[1, 1, 1.3]], 2, return_distance=False)
	
	print(result)


def cluster_plot(X, allocation):
	cmap = ['b', 'g', 'r', 'c', 'm', 'y','k']

	labels = np.unique(allocation)
	n = labels.shape[0]

	#plt.subplot(121)
	for i, j in enumerate(labels):
		subX = X[allocation == labels[i]]
		plt.plot(subX[:,0], subX[:,1], cmap[i] + '.')


	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Clustering Results')
	plt.show()


def STSC(X, k):
	n = X.shape[0]
	if n/(4*k) > 50:
		num_of_samples = 50
	elif n/(4*k) < 2:
		num_of_samples = 3
	else:
		num_of_samples = n/(4*k)


	neigh = NearestNeighbors(num_of_samples)
	neigh.fit(X)

		
	result = neigh.kneighbors(X, num_of_samples, return_distance=False)
	import pdb; pdb.set_trace()



	#n = X.shape[0]
	#D = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean', n_jobs=1)
	#sD = np.sort(D, axis=1)

	#print(sD[:,1:20])
	#import pdb; pdb.set_trace()
	#σ_col = np.reshape(sD[:,6], (n, 1))
	#σ_col = np.reshape(sD[:,20], (n, 1))
	#σ_col[σ_col == 0] = 10


	#print(σ_col[σ_col == 0])
	#import pdb; pdb.set_trace()
	#σ_col = σ_col + 0.00001			# add noise in case of 0

	σ2 = σ_col.dot(σ_col.T)

	K = np.exp(-(D*D)/σ2)
	np.fill_diagonal(K,0)

	D_inv = 1.0/np.sqrt(np.sum(K, axis=1))
	Dv = np.outer(D_inv, D_inv)
	DKD = Dv*K

	[U, U_normalized] = L_to_U(DKD, k)
	allocation = KMeans(k).fit_predict(U_normalized)
	return allocation





#X = genfromtxt('../dataset/smiley.csv', delimiter=',')					# k = 3
#X = genfromtxt('../dataset/spiral_arm.csv', delimiter=',')				# k = 3
#X = genfromtxt('../dataset/inner_rings.csv', delimiter=',')				# k = 3
#X = genfromtxt('../dataset/noisy_two_clusters.csv', delimiter=',')				# k = 3
#X = genfromtxt('../dataset/four_lines.csv', delimiter=',')				# k = 4
#X = preprocessing.scale(X)
#
#labels = STSC(X,4)
#cluster_plot(X, labels)






X = genfromtxt('../dataset/breast-cancer.csv', delimiter=',')	# k = 2
Y = genfromtxt('../dataset/breast-cancer-labels.csv', delimiter=',')	# k = 2

#X = genfromtxt('../dataset/wine.csv', delimiter=',')	# k = 2
#Y = genfromtxt('../dataset/wine_label.csv', delimiter=',')	# k = 2

X = preprocessing.scale(X)
labels = STSC(X,2)
#labels = spectral(X, 1.0, 2)

print(labels)
nmi = normalized_mutual_info_score(labels, Y)
print(nmi)






#samples = [[0, 0, 2], [1, 0, 1], [0, 0, 1], [1,1,1]]
#
#neigh = NearestNeighbors(2)
#neigh.fit(samples)
#result = neigh.kneighbors([[1, 1, 1.3], [0, 0, 1.9]], 2, return_distance=False)
#
#print(result)

