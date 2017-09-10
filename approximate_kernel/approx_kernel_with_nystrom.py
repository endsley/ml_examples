#!/usr/bin/env python

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
import sklearn.metrics
import time 

#	This code demonstrate how using Nystrom could speed up the
#	Calculation of a Kernel

#	K(x, y) = exp(-gamma ||x-y||^2)
#	sigma = sqrt( 1/(2*gamma) )
#	gamma = 1/(2*sigma^2)

num_of_samples = 14000
X = np.random.random((num_of_samples,5))
sampling_percentage = 0.05





start_time = time.time() 
RFF = RBFSampler(gamma=1,n_components= int(num_of_samples*sampling_percentage))
V = RFF.fit_transform(X)
RFF_estimated_kernel = V.dot(V.T)
print("--- RFF Time : %s seconds ---" % (time.time() - start_time))




start_time = time.time() 
N = Nystroem(gamma=1,n_components= int(num_of_samples*sampling_percentage))
V = N.fit_transform(X)
estimated_kernel = V.dot(V.T)
print("--- Nystrom Time : %s seconds ---" % (time.time() - start_time))


start_time = time.time() 
real_kernel = sklearn.metrics.pairwise.rbf_kernel(X, gamma=1)
print("--- Real Time : %s seconds ---" % (time.time() - start_time))



print estimated_kernel[0:5, 0:5]
print '\n\n'
print real_kernel[0:5, 0:5]
print '\n\n'
print RFF_estimated_kernel[0:5, 0:5]



