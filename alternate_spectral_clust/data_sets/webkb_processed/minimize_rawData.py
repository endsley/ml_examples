#!/usr/bin/python

import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing

data = genfromtxt('webkbRaw_word.csv', delimiter=',')
data = preprocessing.scale(data)


np.savetxt('min_words.csv', data, delimiter=',', fmt='%f')
#np.savetxt('min_words.csv', data, delimiter=',', fmt='%d')


import pdb; pdb.set_trace()
