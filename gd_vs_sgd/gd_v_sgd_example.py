#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean
import random

Φ = array([	[1,1],
			[2,1],
			[2,1],
			[3,1]])

n = Φ.shape[0]

y = array([	[1],
			[1],
			[2],
			[2]])

#	The function we are trying to minimize is
#	$$\min_w \; f(x) = \frac{1}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i)^2$$


def f(w):
	fₒ = 0						# function output
	for ϕᵢ, yᵢ in zip(Φ,y):
		ϕᵢ = np.reshape(ϕᵢ, (2,1))
		fₒ += (w.T.dot(ϕᵢ) - yᵢ)**2
	
	return ((1/n)*fₒ).item()
	

#	The equation for the gradient is 
#	$$f'(x) = \frac{2}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i) \phi(x_i)$$

w = array([[0],[1]]) 
η = 0.01

def fᑊ(w):	# Gradient for GD
	ᐁf = np.zeros((2,1))
	for ϕᵢ, yᵢ in zip(Φ,y):
		ϕᵢ = np.reshape(ϕᵢ, (2,1))
		ᐁf += (w.T.dot(ϕᵢ) - yᵢ)*ϕᵢ
	return (2/n)*ᐁf

gd_list = []
for i in range(2000):
	w = w - η*fᑊ(w)				# gradient descent update w
	gd_list.append(f(w))


#	Solving with SGD
wˢ = array([[0],[1]]) 
η = 0.0005

def fˢᑊ(w):	# Gradient for SGD
	# randomly pick a sample
	ϕᵢ, yᵢ = random.choice(list(zip(Φ, y.ravel())))
	ϕᵢ = np.reshape(ϕᵢ, (2,1))
	return 2*(w.T.dot(ϕᵢ) - yᵢ)*ϕᵢ


sgd_list = []
for i in range(60000):
	wˢ = wˢ - η*fˢᑊ(wˢ)
	sgd_list.append(f(wˢ))





print('Best GD w = \n', w)
print('Best SGD w = \n', wˢ)
print('GD Predictions: \n', Φ.dot(w))
print('SGD Predictions: \n', Φ.dot(wˢ))

# Get the points for the best fit line
xp = np.linspace(0,4,10)
fₓ = w[0]*xp + w[1]
fˢ = wˢ[0]*xp + wˢ[1]

#	Draw the best fit line and the data out
plt.figure(figsize=(6,6))
plt.subplot(221)
plt.scatter(Φ[:,0], y, color='red')
plt.plot(xp, fₓ)
plt.title('GD Result')
plt.xlim(0,3)	# Show this region along x-axis
plt.ylim(0,3)	# Show this region along y-axis
#
# Display the error over GD
plt.subplot(222)
plt.plot(gd_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.ylim(0,0.5)	# Show this region along y-axis


#	Draw the best fit line and the data out
plt.subplot(223)
plt.scatter(Φ[:,0], y, color='red')
plt.plot(xp, fˢ)
plt.title('SGD Result')
plt.xlim(0,3)	# Show this region along x-axis
plt.ylim(0,3)	# Show this region along y-axis
#
# Display the error over SGD
plt.subplot(224)
plt.plot(sgd_list)
plt.title('MSE over SGD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.ylim(0,0.5)	# Show this region along y-axis



plt.tight_layout()
plt.show()

