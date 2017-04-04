#!/usr/bin/python

import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.decomposition import PCA


data = genfromtxt('webkbRaw_word.csv', delimiter=',')
data = preprocessing.scale(data)
pca = PCA(n_components=0.70)
data = pca.fit_transform(data)

np.savetxt('min_words_70.csv', data, delimiter=',', fmt='%f')
#np.savetxt('min_words.csv', data, delimiter=',', fmt='%d')


import pdb; pdb.set_trace()
