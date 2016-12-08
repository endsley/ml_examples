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
import matplotlib.pyplot as plt


data = genfromtxt('data_sets/facial_960.csv', delimiter=',')		
#data = genfromtxt('data_sets/facial.csv', delimiter=',')		
label = genfromtxt('data_sets/facial_true_labels.csv', delimiter=',')		
sunglass_label = genfromtxt('data_sets/facial_sunglasses_labels.csv', delimiter=',')		
original_Y = genfromtxt('data_sets/facial_original_Y.csv', delimiter=',')		
name_file = open('data_sets/facial_names.csv', 'r')

names = np.array(name_file.readlines())
name_file.close()


ASC = alt_spectral_clust(data)
db = ASC.db

#ASC.set_values('q',5)
#ASC.set_values('C_num',4)
#ASC.set_values('sigma',20)
#ASC.set_values('kernel_type','Gaussian Kernel')
#ASC.run()
#original = db['allocation']


db['Y_matrix'] = original_Y
db['U_matrix'] = original_Y
db['prev_clust'] = 1



##for k in range(10):
#
ASC.set_values('q',10)
ASC.set_values('sigma',20)
ASC.set_values('C_num',4)
ASC.set_values('lambda',1)
start_time = time.time() 
ASC.run()
print("--- %s seconds ---" % (time.time() - start_time))
alternative = db['allocation']

##print "NMI Against Alternative : " , normalized_mutual_info_score(original,alternative)
##print "NMI Against Sunglasses label : " , normalized_mutual_info_score(sunglass_label,alternative)
#
#	#db['Y_matrix'] = orig_Y_matrix
#
#
#print names[alternative == 1].shape
#print names[alternative == 2].shape
#print names[alternative == 3].shape
#print names[alternative == 4].shape
#
##for m in names[original== 1]: print m, 
for m in names[alternative == 1]: print m, 
##for m in names[alternative == 2]: print m, 
#
#
import pdb; pdb.set_trace()
