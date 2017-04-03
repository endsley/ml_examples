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

 
k = 4
X = genfromtxt('dataset/min_words.csv', delimiter=',')
univ_label = genfromtxt('dataset/webkbRaw_label_univ.csv', delimiter=',') 
topic_label = genfromtxt('dataset/webkbRaw_label_topic.csv', delimiter=',') 


clf = KMeans(n_clusters=k)
allocation = clf.fit_predict(X)
kmeans_nmi = normalized_mutual_info_score(allocation, univ_label)
print "K means : " , kmeans_nmi



d_matrix = sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean')
sigma = np.median(d_matrix)
Gamma = 1/(2*np.power(sigma,2))


clf = SpectralClustering(n_clusters=k, gamma=Gamma)
allocation = clf.fit_predict(X)
spectral_nmi = normalized_mutual_info_score(allocation, univ_label)
print 'Spectral Clustering : ' , spectral_nmi


gmm = GMM(n_components=k)
gmm.fit(X)
allocation = gmm.predict(X)
gmm_nmi = normalized_mutual_info_score(allocation, univ_label)
print 'GMM : ' , gmm_nmi


##import pdb; pdb.set_trace()
#p = [0.3,0.4,0.5,0.6,0.7]   #,0.8,0.9,1,1.1,1.2,1.3,1.4
#drc_nmi = 0
#for m in p:
result = drc(X, k, 1*Gamma, 1.1)		# 1.1 turned out to be the best ratio
allocation = result['allocation']
drc_nmi = normalized_mutual_info_score(allocation, univ_label)
topic_nmi_new = normalized_mutual_info_score(allocation, topic_label)
print 'DRC : '
print '\tUniv : ' , drc_nmi
print '\tTopic: ' , topic_nmi_new

#print 'DRC : ' , m, ' , ' , drc_nmi_new
#if(drc_nmi < drc_nmi_new):
#	drc_nmi = drc_nmi_new
#print 'Dimension :\n' , result['L']


objects = ('K means', 'Spectral', 'GMM', 'DRC')
y_pos = np.arange(len(objects))
performance = [kmeans_nmi, spectral_nmi, gmm_nmi, drc_nmi]
plt.bar(y_pos, performance, align='center', alpha=0.5)
matplotlib.rc('xtick', labelsize=10) 
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('NMI')
plt.title('WebKB Clustering Results against Truth NMI')
info_text = 'Data size : ' + str(X.shape) + '\n'
info_text += 'NMI\nKmeans : ' + str(kmeans_nmi) + '\nSpectral : ' + str(spectral_nmi) 
info_text += '\nGMM : ' + str(gmm_nmi) + '\nDRC : ' + str(drc_nmi)
info_text += '\nSuggested Dimension : ' + str(result['L'].shape[1])
plt.text(1, 0.1, info_text, style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}) 
plt.show()

import pdb; pdb.set_trace()

#plt.text(2, 0.5, info_text, style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}) 
