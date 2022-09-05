#!/usr/bin/env python

#	This examples shows how we can approximate the eigenvectors of a kernel matrix by combining random SVD and nystrom

#	**Method**
#	1. We first subsample p columns, within these p columns, we pick a smaller q columns (p >> q) and use the q columns as L for nystrom
#	2. We find the eigenvector from the q columns to approximate the eigenvectors for p x p matrix as V1
#	3. We next use V1 as a projection matrix for random svd to refine V1 into a better version V2
#	4. We then use V2 (better approximated) again to approximate the eigenvector of the entire kernel matrix K


import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.utils import shuffle
from sklearn.kernel_approximation import Nystroem
from tools import *

#	Initialize all the setting
X = csv_load('../dataset/wine.csv', shuffle_samples=True)
p = 60				
q = 30				
n = X.shape[0]		# number of total samples
γ = get_rbf_γ(X)	# γ used for the gaussian kerenl


#	Use Nystrom to approximate the initial V1
Xa = X[0:q, :]	
Xb = X[0:p, :]


L = sklearn.metrics.pairwise.rbf_kernel(Xb, Y=Xa, gamma=γ)
A = L[0:q,:]
[σs,V] = np.linalg.eig(A)
V = V[:,0:10] # only keeping the largest eigenvectors
Σ = np.diag(1/(np.sqrt(σs[0:10])))
Φ = L.dot(V).dot(Σ)
ǩ = Φ.dot(Φ.T)


