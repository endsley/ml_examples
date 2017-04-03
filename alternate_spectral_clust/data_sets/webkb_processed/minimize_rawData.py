#!/usr/bin/python

import numpy as np
from numpy import genfromtxt

data = genfromtxt('webkbRaw_word.csv', delimiter=',')
np.savetxt('min_words.csv', data, delimiter=',', fmt='%d')

import pdb; pdb.set_trace()
