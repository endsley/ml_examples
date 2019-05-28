#!/usr/bin/env python

from sympy import *
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


def gen_poly_feature(X, order):
	if order != 2 and order != 3: return

	pf = sklearn.preprocessing.PolynomialFeatures(order)
	X_tr = pf.fit_transform(X)
	d =  X_tr.shape[1]

	print(pf.fit_transform(X))
	mask = np.zeros((1,d))
	fname = pf.get_feature_names()


	if order == 2:
		count1 = 0
		count2 = 0

		for counter, m in enumerate(fname):	# For 3rd order
			if m == '1':
				mask[0][counter] = 1
				print('1 : ', m)
			elif m.find('^') != -1:
				mask[0][counter] = np.sqrt(1)
				print('1 : ' , m)
				count1 += 1
			elif m.find(' ') != -1:
				mask[0][counter] = np.sqrt(2)
				print('2 : ' , m)
				count2 += 1
			elif m.count('x') == 1:
				mask[0][counter] = np.sqrt(2)
				print('2 : ' , m)
				count1 += 1
			else:
				import pdb; pdb.set_trace()
				print(m)
		print('mask : ', mask)
		print('count 1 : %d'%count1)
		print('count 2 : %d'%count2)
		print('num of terms : %d'%len(pf.get_feature_names()))

		#xout = X_tr*mask
		#print(xout.dot(xout.T))
		import pdb; pdb.set_trace()

	elif order == 3:
		count3 = 0
		count6 = 0
		for counter, m in enumerate(fname):	# For 3rd order
			if m == '1':
				mask[0][counter] = 1
				print('1 : ', m)
			elif m.find('^') != -1:
				mask[0][counter] = np.sqrt(3)
				print('3 : ' , m)
				count3 += 1
			elif m.find(' ') != -1:
				mask[0][counter] = np.sqrt(6)
				print('6 : ' , m)
				count6 += 1
			elif m.count('x') == 1:
				mask[0][counter] = np.sqrt(3)
				print('3 : ' , m)
				count3 += 1
			else:
				import pdb; pdb.set_trace()
				print(m)
		
		print('mask : ', mask)
		print('count 3 : %d'%count3)
		print('count 6 : %d'%count6)
		print('num of terms : %d'%len(pf.get_feature_names()))

		xout = X_tr*mask
		print(xout.dot(xout.T))






init_printing()

x0 = Symbol('x0');y0 = Symbol('y0')
x1 = Symbol('x1');y1 = Symbol('y1')
x2 = Symbol('x2');y2 = Symbol('y2')
x3 = Symbol('x3');y3 = Symbol('y3')
x4 = Symbol('x4');y4 = Symbol('y4')


#equation = (x1**2+x2**2+x3**2+x1+x2+x3+x1*x2+x1*x3+x2*x3+1)**2
equation = (x1+x2+x3+1)**4
out = expand(equation)
print('num of terms : %d'%len(out.args))
#pprint(out.args)
for m in out.args:
	print(m)

import pdb; pdb.set_trace()
#print(latex(out))
#print(srepr(out))
#print(out._sorted_args)

#print(out.args)
#out._from_args
#print(out.subs({x1:1, x2: 1}))
#

#X = np.array([[1,2,1,1,1],[0,2,3,1,2]])
#X = np.array([[1,2,1],[0,1,1]])
#gen_poly_feature(X, 2)
#
#K = sklearn.metrics.pairwise.polynomial_kernel(X, degree=2, coef0=1)
#
#print(K)

#pf = sklearn.preprocessing.PolynomialFeatures(3)
#print(pf.fit_transform(np.array([[1,2,1,1,1]])))
#print( pf.get_feature_names() )

#for m in pf.get_feature_names():	# For 2nd order
#	if m == '1':
#		print('1 : ', m)
#	elif m.find('^') != -1:
#		print('1 : ' , m)
#	elif m.find(' ') != -1:
#		print('2 : ' , m)
#	elif m.find('*') == -1:
#		print('3 : ' , m)
#	else:
#		print(m)

#count3 = 0
#count6 = 0
#for m in pf.get_feature_names():	# For 3rd order
#	if m == '1':
#		print('1 : ', m)
#	elif m.find('^') != -1:
#		print('3 : ' , m)
#		count3 += 1
#	elif m.find(' ') != -1:
#		print('6 : ' , m)
#		count6 += 1
#	elif m.count('x') == 1:
#		print('3 : ' , m)
#		count3 += 1
#	else:
#		import pdb; pdb.set_trace()
#		print(m)
#
#print('count 3 : %d'%count3)
#print('count 6 : %d'%count6)
#print('num of terms : %d'%len(pf.get_feature_names()))
#import pdb; pdb.set_trace()

#['1', 'x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']
#>>> pf.fit_transform(np.array([[1,2,3,4,5,6]]))
#array([[ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  1.,  2.,  3.,  4.,  5.,  6.,
#         4.,  6.,  8., 10., 12.,  9., 12., 15., 18., 16., 20., 24., 25.,
#        30., 36.]])
#>>> pf.get_feature_names()
#['1', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x0 x4', 'x0 x5', 'x1^2', 'x1 x2', 'x1 x3', 'x1 x4', 'x1 x5', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x3^2', 'x3 x4', 'x3 x5', 'x4^2', 'x4 x5', 'x5^2']
#>>> names = pf.get_feature_names()






## auto sklearn
#import numpy as np
#import matplotlib.pyplot as plt
#import autosklearn.classification
#import sklearn.model_selection
#import sklearn.datasets
#import sklearn.metrics
#
#
#X1 = np.random.randn(100,2) + np.array([6,8])
#X2 = np.random.randn(100,2) + np.array([-6,8])
#X3 = np.random.randn(100,2) + np.array([0,-8])
#
#plt.subplot(111)
#plt.plot(X1[:,0], X1[:,1], 'b.')
#plt.plot(X2[:,0], X2[:,1], 'g.')
#plt.plot(X3[:,0], X3[:,1], 'r.')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Linearly Separable Data')
#plt.show()
#
#import pdb; pdb.set_trace()


#(Pdb) out.is_polynomial
#<bound method Expr.is_polynomial of x**2 + 2*x*y + 2*x + y**2 + 2*y + 1>
#
