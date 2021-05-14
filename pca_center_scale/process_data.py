#!/usr/bin/env python

from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from numpy import genfromtxt
import pandas as pd
import sys

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


def pca_center_scale(data, keep_variance): #keep_variance 0 to 1
	#	Assuming the data with each row as a single sample
	data = data.astype(float)
	pca = PCA(n_components=keep_variance)
	X_pca = pca.fit_transform(data)
	X = preprocessing.scale(X_pca)

	return X

#np.nanmean and np.nanstd		should give the same stuff
def center_scale_with_missing_data(X, replace_nan_with_0=False): 
	d = X.shape[1]
	ignore_column_with_0_σ = []
	for i in range(d):
		x = X[:,i]
		ẋ = x[np.invert(np.isnan(x))]
		ẋ = ẋ - np.mean(ẋ)
		σ = np.std(ẋ)

		if σ < 0.00001:
			ignore_column_with_0_σ.append(i)		# add this to delete list
		else:
			X[np.invert(np.isnan(x)), i] = ẋ/σ


	for i in ignore_column_with_0_σ:
		X = np.delete(X, i , axis=1)	# # delete columns with σ=0

	if replace_nan_with_0:
		X = np.nan_to_num(X)

	return X, ignore_column_with_0_σ



df = pd.read_csv ('chem.exposures.csv')
X = center_scale_with_missing_data(df.to_numpy(), replace_nan_with_0=True)
import pdb; pdb.set_trace()



#data = genfromtxt('mnist.csv', delimiter=',')
#X = center_scale_PCA(data, 0.9)
#np.savetxt('mnist_pca_90.csv', X, delimiter=',', fmt='%.4f') 
#print(X.mean(axis=0))
#print(X.std(axis=0))

