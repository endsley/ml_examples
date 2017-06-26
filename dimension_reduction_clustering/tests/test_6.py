#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
import sklearn.metrics
from drc import *
import pickle
import matplotlib
from numpy import genfromtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from PIL import Image
colors = matplotlib.colors.cnames
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

 


#	Generate Data
im = Image.open("dataset/Flower2.png")

rgb_im = im.convert('RGB')
Img_3d_array = np.asarray(rgb_im)
data = np.empty((0, 3))
data_dic = {}

for i in range(Img_3d_array.shape[0]):
	for j in range(Img_3d_array.shape[0]):
		data_dic[ str(Img_3d_array[i,j]) ] = Img_3d_array[i,j]


for i,j in data_dic.items():
	data = np.vstack((data, j))

X = preprocessing.scale(data)

#-------------------------------------------------


k = 3

#univ_label = genfromtxt('dataset/webkbRaw_label_univ.csv', delimiter=',') 
#topic_label = genfromtxt('dataset/webkbRaw_label_topic.csv', delimiter=',') 


#clf = KMeans(n_clusters=k)
#allocation = clf.fit_predict(X)




d_matrix = sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean')
sigma = np.median(d_matrix)
Gamma = 1/(2*np.power(sigma,2))


#clf = SpectralClustering(n_clusters=k, gamma=Gamma)
#allocation = clf.fit_predict(X)
#spectral_nmi = normalized_mutual_info_score(allocation, univ_label)
#print 'Spectral Clustering : ' , spectral_nmi

#gmm = GMM(n_components=k)
#gmm.fit(X)
#allocation = gmm.predict(X)



result = drc(X, k, 0.07*Gamma, Const=50.1)		# 1.1 turned out to be the best ratio
allocation = result['allocation']


if True:	#	Plot clustering results
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	Uq_a = np.unique(allocation)
	
	group1 = X[allocation == Uq_a[0]]
	group2 = X[allocation == Uq_a[1]]
	group3 = X[allocation == Uq_a[2]]
	
	ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
	ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
	ax.scatter(group3[:,0], group3[:,1], group3[:,2], c='g', marker='x')
	ax.set_xlabel('Feature 1')
	ax.set_ylabel('Feature 2')
	ax.set_zlabel('Feature 3')
	ax.set_title('Original Clustering')
	
	plt.show()









#
#objects = ('K means', 'Spectral', 'GMM', 'DRC')
#y_pos = np.arange(len(objects))
#performance = [kmeans_nmi, spectral_nmi, gmm_nmi, drc_nmi]
#plt.bar(y_pos, performance, align='center', alpha=0.5)
#matplotlib.rc('xtick', labelsize=10) 
#plt.xticks(y_pos, objects, rotation='vertical')
#plt.ylabel('NMI')
#plt.title('WebKB Clustering Results against Truth NMI')
#info_text = 'Data size : ' + str(X.shape) + '\n'
#info_text += 'NMI\nKmeans : ' + str(kmeans_nmi) + '\nSpectral : ' + str(spectral_nmi) 
#info_text += '\nGMM : ' + str(gmm_nmi) + '\nDRC : ' + str(drc_nmi)
#info_text += '\nSuggested Dimension : ' + str(result['L'].shape[1])
#plt.text(1, 0.1, info_text, style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}) 
#plt.show()
#
#import pdb; pdb.set_trace()
#
##plt.text(2, 0.5, info_text, style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}) 
