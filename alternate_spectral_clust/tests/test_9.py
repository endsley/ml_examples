#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
from cost_function import *


data = genfromtxt('data_sets/moon.csv', delimiter=',')		

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',2)
ASC.set_values('C_num',2)
ASC.set_values('sigma',2)
ASC.set_values('kernel_type','Gaussian Kernel')

ASC.run()
a = db['allocation']


ASC.set_values('sigma',0.3)
ASC.run()
b = db['allocation']

import pdb; pdb.set_trace()
