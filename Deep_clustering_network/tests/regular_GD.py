#!/usr/bin/python


import torch
from torch.autograd import Variable
import numpy as np
import numpy.matlib
from numpy import genfromtxt
import sklearn.metrics

np.set_printoptions(precision=5)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

#	Numpy implementation
data = genfromtxt('datasets/data_4.csv', delimiter=',')
N = data.shape[0]
d = data.shape[1]
k = 4
lmda = 1*np.ones((N,1))

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma = np.median(d_matrix)
gamma = 1/(2*np.power(sigma,2))
learning_rate = 0.1

K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=gamma)
Y = np.ones((N,k))
offset = np.sum(Y, axis=1) - 1
L = -K

def cost(L, Y, lmda):
	cluster_quality = np.sum((Y.dot(Y.T))*L)
	offset = np.sum(np.abs(Y), axis=1) - 1
	offset = offset*offset
	regularizer = lmda.T.dot(offset)

	cost = cluster_quality + regularizer
	return cost

def gradient(L, Y, lmda):
	offset = np.sum(np.abs(Y), axis=1) - 1
	offset = offset.reshape((40,1))
	alpha = 2*lmda*offset

	grad = 2*L.dot(Y) + np.matlib.repmat(alpha, 1, k)*np.sign(Y)
	return grad
	
def get_error(Y, lmda):
	reg = (np.sum(np.abs(Y), axis=1) - 1)
	reg = reg*reg
	reg = reg.reshape((40,1))
	return [np.sum(reg), reg.T.dot(lmda)]

	#print 'residual : ' , np.sum(reg), reg.T.dot(lmda)

for n in range(100):
	for m in range(100):
		grad = gradient(L, Y, lmda)
	
		print 'Before : ', cost(L, Y, lmda)
		Y -= 0.0001*grad
		print 'After : ', cost(L, Y, lmda)
		print 'error : ' , get_error(Y, lmda)
	
	import pdb; pdb.set_trace()
	[error, reg] = get_error(Y, lmda)
	lmda += 0.1*reg



import pdb; pdb.set_trace()






##	Pytoch implementation
#data = genfromtxt('datasets/data_4.csv', delimiter=',')
#N = data.shape[0]
#d = data.shape[1]
#k = 4
#lmda_hold = 0.1*np.ones((N,1))
#
#d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
#sigma = np.median(d_matrix)
#gamma = 1/(2*np.power(sigma,2))
#
#
#dtype = torch.FloatTensor
#learning_rate = 0.1
#
#K = sklearn.metrics.pairwise.rbf_kernel(data, gamma=gamma)
#K = torch.from_numpy(K)
#K = Variable(K.type(dtype), requires_grad=False)
#
#Y = torch.from_numpy(np.ones((N,k)))
#Y = Variable(Y.type(dtype), requires_grad=True)
#
#lmda = torch.from_numpy(lmda_hold)
#lmda = Variable(lmda.type(dtype), requires_grad=False)
#
#for m in range(30):
#	cluster_quality = (torch.mm(Y, Y.transpose(0,1))*K).sum()
#	offset = torch.abs(Y).sum(dim=1) - 1
#	offset = offset*offset
#	regularizer = torch.mm(lmda.transpose(0,1), offset)
#
#	loss = cluster_quality - 100*regularizer
#	loss.backward()
#
#	Y.data += learning_rate*Y.grad.data
#	Y.grad.data.zero_()
#
#	print loss.data[0] , cluster_quality.data[0], regularizer.data.numpy()
#import pdb; pdb.set_trace()
#
#
