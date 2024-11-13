#!/usr/bin/env python

import numpy as np
import sys
from numpy.linalg import eig
from numpy.linalg import svd

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


#	This code tests the validity of mercer's theorem
def pretty_np_array(m, front_tab='', verticalize=False, title=None, auto_print=False):
	m = str(m)

	if verticalize:
		if len(m.shape) == 1:
			m = np.atleast_2d(m).T

	out_str = front_tab + str(m).replace('\n ','\n' + front_tab).replace('[[','[').replace(']]',']') + '\n'
	out_str = str(out_str).replace('.]',']')

	if type(title).__name__ == 'str':
		L1 = out_str.split('\n')
		L1_max_width = len(max(L1, key=len))
		t1 = str.center(title, L1_max_width)
		out_str = t1 + '\n' + out_str

	if auto_print: print(out_str)
	else: return out_str


def block_two_string_concatenate(str1, str2, spacing='\t', add_titles=[], auto_print=False):
	str1 = str(str1)
	str2 = str(str2)

	L1 = str1.split('\n')
	L2 = str2.strip().split('\n')

	if len(L1) > len(L2):
		Δ = len(L1) - len(L2)
		for ι in range(Δ):
			L2.append('\n')

	if len(add_titles) == 2:
		L1_max_width = len(max(L1, key=len))
		L2_max_width = len(max(L2, key=len))
		t1 = str.center(add_titles[0], L1_max_width)
		t2 = str.center(add_titles[1], L2_max_width)
		L1.insert(0,t1)
		L2.insert(0,t2)

	max_width = len(max(L1, key=len))
	outS = ''
	for l1, l2 in zip(L1,L2):
		outS += ('%-' + str(max_width) + 's' + spacing + l2 + '\n') % l1

	if auto_print: print(outS)
	else: return outS


def polyK(X):	# obtain the poly
	K = np.zeros((len(X), len(X)))

	for i, x in enumerate(X):
		for j, y in enumerate(X):
			xi = x.reshape(len(x), 1)
			yi = y.reshape(len(y), 1)
			K[i,j] = (xi.T.dot(yi) + 1)**2

	return K

def Φ(X):
	fm = np.empty((6, 0))
	c = np.sqrt(2)

	for x in X:
		# feature map of polynomial [x₁ᒾ,  ᴄ x₁x₂, ᴄ x₁, c x₂, x₂ᒾ, 1]
		φ = np.array([[x[0]*x[0], c*x[0]*x[1], c*x[0], c*x[1], x[1]*x[1], 1]]).T
		fm = np.hstack((fm, φ))

	return fm.T



X = np.array([[1,1], [2,2], [3,3]])
n = X.shape[0]


Q = Φ(X)
K = polyK(X)
#Tn = (1/n)*Q.T.dot(Q)
Tn = Q.T.dot(Q)

[Dk, Vk] = eig(K)				
[Dq, Vq] = eig(Tn)			# σ of Tn * N = σ of K

σ = Dq[0:3]
V = Vq[:,0:3]
#nM = (V.dot(np.diag(σ)).dot(V.T))
nM = (V.dot(V.T))
aK = Q.dot(nM).dot(Q.T)

import pdb; pdb.set_trace()
#eK = pretty_np_array(Dk, front_tab='', title='Eig Values of K', auto_print=False)
#eQ = pretty_np_array(Dq, front_tab='', title='Eig Values of Q', auto_print=False)
#block_two_string_concatenate(eK, eQ, spacing='\t', add_titles=[], auto_print=True)
#
#eigK = pretty_np_array(Vk, front_tab='', title='Eig of K', auto_print=False)
#eigQ = pretty_np_array(Vq, front_tab='', title='Eig of Q', auto_print=False)
#block_two_string_concatenate(eigK, eigQ, spacing='\t', add_titles=[], auto_print=True)
#
#eigFun_from__eigV_of_K = (1/np.sqrt(Dk))*((Q.T.dot(Vk)))
#eigFun = Vq[:,0:3]
#
#eigFun_from__eigV_of_K = pretty_np_array(eigFun_from__eigV_of_K, front_tab='', title='Computed eig Function', auto_print=False)
#eigFun = pretty_np_array(eigFun, front_tab='', title='True Eigen Function', auto_print=False)
#block_two_string_concatenate(eigFun_from__eigV_of_K, eigFun, spacing='\t', add_titles=[], auto_print=True)

