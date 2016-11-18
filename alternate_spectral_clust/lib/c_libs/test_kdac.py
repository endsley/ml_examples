#! /usr/bin/env python

import Nice4Py
import numpy as np
from numpy import genfromtxt

data = genfromtxt('data_gaussian_2.csv', delimiter=',')
input_dim = data.shape
kdac = Nice4Py.KDAC()

# Kernel type can be Gaussian|Linear|Polynomial
params = {'c':2, 'q':2, 'kernel':'Gaussian', 'lambda':1.0, 'sigma':1.0}
kdac.SetupParams(params)
output = np.empty((input_dim[0], 1))
print data
print "=============="
kdac.Fit(data, input_dim[0], input_dim[1])
kdac.Predict(output, input_dim[0], 1)
print output
print "=============="
kdac.Fit()
kdac.Predict(output, input_dim[0], 1)
print output
print "=============="

# If you want to get matrix U
# the syntax is as below
# kdac.GetU(output, <row number of U>, <row number of V>)
# Getting Matrix W uses same syntax

# Another Getters GetQ() GetD() GetN(), No argument needed
q = kdac.GetQ();
d = kdac.GetD();
n = kdac.GetN();
print q
print d
print n
