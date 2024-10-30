#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

X = np.array([	[0,2],
				[0,1],
				[1,0],
				[3,4],
				[5,2],
				[5,3]])

y = np.array([[-1],[-1],[-1],[1],[1],[1]])

w = np.array([[0],[1]])
η = 0.01

#	$$\frac{dL}{dw} = 2w - X^{\top} y	$$

def dw(X, y, w):
	return 2*w - X.T.dot(y)

def db(y):
	return np.sum(y)

def GD(X, y, w):
	for i in range(100):
		w = w - η*dw(X,y,w)
	return w

w = GD(X, y, w)

#	Note that for this example, I already knew which points
#	Are the support vectors (X[3], X[0]) 
#	In actual problems, you would need to project the data 
#	Down to w and then figure out the mid-point between the 2 clusters

w = w/np.linalg.norm(w)
a = (X[3].dot(w))
c = (X[0].dot(w))

b = -(a + c)/2
mid = -b*w

#	drawing the separating line
w1 = w[0,0]
w2 = w[1,0]
line_x = np.arange(1, 3, 0.1)
line_y = (-w1/w2)*line_x - (1/w2)*b

#	Drawing the w vector
wx = np.array([[0],[0]])
wy = 6*w
wLine = np.hstack((wx,wy))


plt.scatter(X[:,0], X[:,1], color='blue', marker='x')
plt.scatter(mid[0], mid[1], color='blue', marker='o')
plt.plot(line_x, line_y, color='red')
plt.plot(wLine[0], wLine[1], color='green')
plt.title('Simplified SVM', fontsize=12)
plt.show()


