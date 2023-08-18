#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 

X = np.array([	[0,0],
				[1,0],
				[1,1],
				[2,1],
				[2,2]])

#y = np.array([0,1,2,5,6])
y = np.array([0,1,2,3,4])
η = 0.002

Wᑊ = np.random.randn(2,2)
Wᒾ = np.random.randn(2,1)

def f(x):
	h = np.maximum(0, Wᑊ.T.dot(x))
	return Wᒾ.T.dot(h)


errors = []
for s in range(40):
	dWᒾ = 0
	for i, j in enumerate(y):
		x = np.reshape(X[i], (2,1))
		h = np.maximum(0, Wᑊ.T.dot(x))
		dWᒾ += (f(x) - y[i])*h
	
	Wᒾ = Wᒾ - η*dWᒾ

	dWᑊ = 0
	for i, j in enumerate(y):
		x = np.reshape(X[i], (2,1))
		h = np.maximum(0, Wᑊ.T.dot(x))
		〡 = np.heaviside(h,0)

		#dWᑊ += (f(x) - y[i])*(x.dot((Wᒾ*〡).T))
		dWᑊ += (f(x) - y[i])*np.kron((Wᒾ*〡).T, x)

	Wᑊ = Wᑊ - η*dWᑊ
	e = np.sum(np.square(f(X.T) - y))
	errors.append(e)


print(f(X.T))

plt.plot(range(len(errors)), errors, label='Line')
plt.show()






#from sklearn.neural_network import MLPRegressor
#regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(2)).fit(X, y)
#print(regr.predict(X))
