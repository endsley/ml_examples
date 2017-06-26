#!/usr/bin/python


import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import scipy.io as sio



X = np.r_[np.random.randn(30, 1), np.random.randn(30, 1) + 10]
print X.shape

# Fit a Dirichlet process mixture of Gaussians using five components
dpgmm = mixture.DPGMM(n_components=4, covariance_type='diag')
dpgmm.fit(X)

Y = dpgmm.predict(X)
print Y

plt.hist(X)
plt.show()
