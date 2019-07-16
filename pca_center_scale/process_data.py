#!/usr/bin/env python

from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from numpy import genfromtxt

data = genfromtxt('mnist.csv', delimiter=',')

def pca_center_scale(data, keep_variance): #keep_variance 0 to 1
	#	Assuming the data with each row as a single sample
	data = data.astype(float)
	pca = PCA(n_components=keep_variance)
	X_pca = pca.fit_transform(data)
	X = preprocessing.scale(X_pca)

	return X

X = center_scale_PCA(data, 0.9)
np.savetxt('mnist_pca_90.csv', X, delimiter=',', fmt='%.4f') 

#print(X.mean(axis=0))
#print(X.std(axis=0))

