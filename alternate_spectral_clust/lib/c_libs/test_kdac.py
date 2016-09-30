#! /usr/bin/env python

import Nice4Py
import numpy as np
from numpy import genfromtxt

data = genfromtxt('data_gaussian_2.csv', delimiter=',')
input_dim = data.shape
i = Nice4Py.PyInterface()
i.Init(data, Nice4Py.ModelType.KDAC, input_dim[0], input_dim[1])
params = {}
i.SetupParams(params)
output = np.empty((input_dim[0], 1))
print data
print "=============="
i.Run(output, input_dim[0], 1)
print output
print "=============="
i.Run(output, input_dim[0], 1)
print output
print "=============="

