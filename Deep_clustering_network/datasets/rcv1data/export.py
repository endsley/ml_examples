#!/usr/bin/python

import numpy as np
import numpy.matlib

X = np.load('datarcv1.npy')
label = np.load('reutersidf.npy')

#X = np.arange(10)
#X = X.reshape((10,1))
#X = np.matlib.repmat(X,1,3)

keep_N = 10000
P = np.random.permutation(X.shape[0])
#print P
out = X[P[0:keep_N], :]
label_10000 = label[P[0:keep_N]]

np.savetxt('rcv_10000.csv', out, delimiter=',', fmt='%d')
np.savetxt('label_10000.csv', label_10000, delimiter=',', fmt='%d')

import pdb; pdb.set_trace()
