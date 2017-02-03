#! /usr/bin/env python

import Nice4Py
import numpy as np
from numpy import genfromtxt

#data = genfromtxt('Four_gaussian_3D.csv', delimiter=',', dtype=np.float32)
data = genfromtxt('data_4.csv', delimiter=',', dtype=np.float32)
input_dim = data.shape
kdac = Nice4Py.KDAC('gpu')		# it could be either cpu or gpu as input

# Kernel type can be Gaussian|Linear|Polynomial
params = {'c':2, 'q':1, 'kernel':'Gaussian', 'lambda':1.0, 'sigma':1.0, 'verbose':1.0}
kdac.SetupParams(params)
output = np.empty((input_dim[0], 1), dtype=np.float32)
#print data
print "=============="
kdac.Fit(data, input_dim[0], input_dim[1])
kdac.Predict(output, input_dim[0], 1)
print output.T
print "=============="
import pdb; pdb.set_trace()

kdac.Fit()
kdac.Predict(output, input_dim[0], 1)
print output.T
print "=============="

## If you want to get matrix U
## the syntax is as below
## kdac.GetU(output, <row number of U>, <row number of V>)
## Getting Matrix W uses same syntax
#
## Another Getters GetQ() GetD() GetN(), No argument needed
#q = kdac.GetQ();
#d = kdac.GetD();
#n = kdac.GetN();
#print q
#print d
#print n
