#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,1],[2,1],[2,2],[3,2]])

alpha = 1
beta = 1
c = 0.2
alpha_beta_list = []

def objective(x,alpha,beta):
	objective_sum = 0
	for n in range(x.shape[0]): 
		objective_sum += (alpha*x[n,0] + beta - x[n,1])*(alpha*x[n,0] + beta - x[n,1])
		
	return objective_sum


	
old_mag = objective(x,alpha,beta)
print 'Initial magnitude : ' , old_mag
result_converged = False
while not result_converged :
	alpha_grad = 0
	beta_grad = 0

	#	Calculate for the derivative
	for n in range(x.shape[0]): 
		alpha_grad = 2*alpha*x[n, 0]*x[n, 0] + 2*beta*x[n,0] - 2*x[n,0]*x[n,1]
		beta_grad  = 2*alpha*x[n, 0] + 2*beta - 2*x[n,1]


		#	Make sure that the constant value c is small enough
		c_small_enough = False
		while not c_small_enough:
			new_alpha = alpha - c*alpha_grad
			new_beta = beta - c*beta_grad
		
			if old_mag < objective(x,new_alpha,new_beta):
				c = 0.90*c
			else: 
				c_small_enough = True
				alpha = new_alpha
				beta = new_beta
				alpha_beta_list.append([alpha,beta])
	
		new_magnitude = objective(x,alpha,beta)	
		magnitude_change = np.abs(old_mag - new_magnitude)
		print new_magnitude
		if magnitude_change < new_magnitude*0.00001:
			result_converged = True
			old_mag = new_magnitude
		else:
			old_mag = new_magnitude


print 'Best fit line : ' , alpha , 'x + (', beta , ') = y' 
print 'Final magnitude : ' , new_magnitude
	


for m in alpha_beta_list:
	x1 = np.arange(0,5)
	y1 = x1*m[0] + m[1]
	plt.plot(x1.tolist(), y1.tolist(), '0.9')

x1 = np.arange(0,5)
y1 = x1*alpha + beta
plt.plot(x1.tolist(), y1.tolist(), 'b')
plt.plot(x[:,0].tolist(), x[:,1].tolist(), 'ro')

plt.axis([0,4,0,5])
plt.show()
