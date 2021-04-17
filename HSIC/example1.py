#!/usr/bin/env python

import numpy as np
from numpy.random import randn
from Dependency.HSIC import HSIC

#	Assume Y is for classification
X = np.vstack((randn(3,2), randn(3,2) + 3))
Y = np.hstack((np.zeros(3), np.ones(3)))
h1 = HSIC(X,Y, X_kernel='Gaussian', Y_kernel='linear', sigma_type='opt', normalize_hsic=True)


#	Assume Y is for regression
Y = 2*X + np.random.randn(6,2)
h2 = HSIC(X,Y, X_kernel='Gaussian', Y_kernel='Gaussian', sigma_type='opt', normalize_hsic=True)

print('HSIC under Classification : %.3f'%h1)
print('HSIC under Regression : %.3f'%h2)
