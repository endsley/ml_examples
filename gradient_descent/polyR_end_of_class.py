#!/usr/bin/env python

import numpy as np
from numpy import ones
from numpy import array
import matplotlib.pyplot as plt
from numpy import mean

Φ = array([	[1, 1,1],
			[4, 2,1],
			[2.25, 1.5,1],
			[9, 3,1]])

n = Φ.shape[0]
d = Φ.shape[1]

y = array([	[1],
			[1],
			[0],
			[2]])

w = array([[1],[1],[1]]) 
η = 0.01

def f(w):
	ε = Φ.dot(w) - y	# error for each sample
	mse = (1/n)*ε.T.dot(ε)	# average error squared
	return mse.item()		# make sure it is scalar
	

#	The equation for the gradient is 
#	$$f'(x) = \frac{2}{n} \; \sum_i^n \; (w^{\top} \phi(x_i) - y_i) \phi(x_i)$$
#	You can rewrite it into compact matrix vector form if you are good, note that $y$ is
#	a column vector of all labels.
#	The derivation can be found here:
#	https://github.khoury.northeastern.edu/chieh/course_page/blob/main/4420/lecture_4/compact_regression_derivative_derivation.pdf
#	$$f'(x) = \frac{2}{n} \Phi^{\top}(\Phi w - y)$$

def fᑊ(w):
	ᐁf = np.zeros((d,1))
	for Φᵢ, yᵢ in zip(Φ,y):
		Φᵢ = np.reshape(Φᵢ, (d,1))
		ᐁf += (w.T.dot(Φᵢ) - yᵢ)*Φᵢ
	return (2/n)*ᐁf

f_value_list = []
for i in range(400):
	w = w - η*fᑊ(w)				# gradient descent update w
	f_value_list.append(f(w))

print('Best w = \n', w)
print('Predictions: \n', Φ.dot(w))

# Get the points for the best fit line
xp = np.linspace(0,4,10)
fₓ = w[0]*xp*xp + w[1]*xp + w[2]

#	Draw the best fit line and the data out
plt.figure(figsize=(6,3))
plt.subplot(121)
plt.scatter(Φ[:,1], y, color='red')
plt.plot(xp, fₓ)
plt.xlim(-1,4)	# Show this region along x-axis
plt.ylim(-1,4)	# Show this region along y-axis
#
# Display the error over GD
plt.subplot(122)
plt.plot(f_value_list)
plt.title('MSE over GD')
plt.xlabel('steps')
plt.ylabel('MSE')
plt.tight_layout()
plt.show()

