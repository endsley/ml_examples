#!/usr/bin/python

import sys
sys.path.append('./lib')
from alt_spectral_clust import *
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy.matlib
import pickle
import sklearn
import time 

#	Min -0.25

#np.set_printoptions(suppress=True)
data = genfromtxt('data_sets/gene_dat.csv', delimiter=',')

ASC = alt_spectral_clust(data)
omg = objective_magnitude
db = ASC.db

ASC.set_values('q',118)
ASC.set_values('C_num',4)
ASC.set_values('lambda',1)
ASC.set_values('kernel_type','Gaussian Kernel')
ASC.set_values('sigma',15)
ASC.run()
a = db['allocation']

#print db['Y_matrix']
start_time = time.time() 
ASC.run()
print("--- %s seconds ---" % (time.time() - start_time))
b = db['allocation']

print "NMI : " , normalized_mutual_info_score(a,b)

#sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean', n_jobs=1, **kwds)

#new_d = db['data'].dot(db['W_matrix'])
#dm = sklearn.metrics.pairwise.pairwise_distances(new_d)
#np.savetxt('original_similarity.txt', db['Kernel_matrix'], fmt='%5.3f', delimiter=',', newline='\n', header='', footer='', comments='# ')
import pdb; pdb.set_trace()
