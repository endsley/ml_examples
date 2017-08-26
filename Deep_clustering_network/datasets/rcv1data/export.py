#!/usr/bin/python

import numpy as np
import numpy.matlib
from numpy import genfromtxt
from PIL import Image
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing

def center_scale_PCA(data, keep_variance):
	#	Assuming the data with each row as a single sample
	data = data.astype(float)
	data = preprocessing.scale(data)
	pca = PCA(n_components=keep_variance)
	data = pca.fit_transform(data)
	return preprocessing.scale(data)



data = genfromtxt('rcv_10000.csv', delimiter=',')
X = center_scale_PCA(data, 0.90)
np.savetxt('rcv_10000_pca.csv', X, delimiter=',', fmt='%d')

import pdb; pdb.set_trace()







##	Sampling 10000 data from the total
#X = np.load('datarcv1.npy')
#label = np.load('reutersidf.npy')
#
##X = np.arange(10)
##X = X.reshape((10,1))
##X = np.matlib.repmat(X,1,3)
#
#keep_N = 10000
#P = np.random.permutation(X.shape[0])
##print P
#out = X[P[0:keep_N], :]
#label_10000 = label[P[0:keep_N]]
#
#np.savetxt('rcv_10000.csv', out, delimiter=',', fmt='%d')
#np.savetxt('label_10000.csv', label_10000, delimiter=',', fmt='%d')
#
#import pdb; pdb.set_trace()
