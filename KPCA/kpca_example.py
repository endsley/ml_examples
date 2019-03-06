#!/usr/bin/env python

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
from sklearn.manifold import SpectralEmbedding
import sklearn.metrics
from sklearn.decomposition import PCA



(X, Y, frac0, frac1) = pickle.load( open( "mydataset.p", "rb" ) )
σ = 0.4
γ = 1.0/(2*σ*σ)

#	PCA
pca = PCA(n_components=1, svd_solver='full')
pca_out = pca.fit_transform(X)                 


#	KPCA
embedding = SpectralEmbedding(n_components=1)
K = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
H = np.eye(800) - (1.0/800)*np.ones((800,800))
K = H.dot(K).dot(H)
Xout = embedding.fit_transform(K)



plt.figure(figsize=(10,3))
plt.subplot(131)
plt.plot(X[Y == 0][:,0], X[Y == 0][:,1], 'go')
plt.plot(X[Y == 1][:,0], X[Y == 1][:,1], 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data in original space')

plt.subplot(132)
plt.plot(pca_out[Y == 0][:,0], np.zeros((400,1)), 'go')
plt.plot(pca_out[Y == 1][:,0], np.zeros((400,1)), 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data After PCA')


plt.subplot(133)
#plt.plot(Xout[Y == 1][:,0], Xout[Y == 1][:,1], 'g.')
#plt.plot(Xout[Y == 0][:,0], Xout[Y == 0][:,1], 'b.')
plt.plot(Xout[Y == 0][:,0], np.zeros((400,1)), 'go')
plt.plot(Xout[Y == 1][:,0], np.zeros((400,1)), 'bx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data After Gaussian KPCA $\sigma=%.1f$'%σ)




plt.subplots_adjust(wspace = 0.2)
plt.tight_layout()
plt.show()

import pdb; pdb.set_trace()

