#!/usr/bin/python

import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import scipy.io as sio


np.set_printoptions(threshold=np.nan)
mat_contents = sio.loadmat('Euclid.mat')

row = mat_contents['Euclid_matrix'][:,100]
  
dpgmm = mixture.DPGMM(n_components=6, covariance_type='diag', alpha=10, n_iter=100,verbose=0, min_covar=0.0001)
dpgmm.fit(row)
Y = dpgmm.predict(row)
print Y

Y_unique = np.unique(Y)
print '\nThe 2 sets'
for point in Y_unique:
	print row[Y == point]

print '\nsettings'
print dpgmm.get_params()
print dpgmm.n_components
print dpgmm.precs_

print row.shape
print Y.shape
print dpgmm.lower_bound(row, Y)


plt.hist(row,100)
plt.show()



