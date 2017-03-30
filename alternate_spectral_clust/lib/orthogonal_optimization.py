#!/usr/bin/python

from numpy import *

class orthogonal_optimization:
	def __init__(self, cost_function, gradient_function):
		self.cost_function = cost_function
		self.gradient_function = gradient_function
		self.x_opt = None
		self.cost_opt = None

	def calc_A(self, x):
		G = self.gradient_function(x)
		A = G.dot(x.T) - x.dot(G.T)
		return A

	def run(self, x_init):
		d = x_init.shape[0]
		self.x_opt = x_init
		I = eye(d)
		max_rep = 200
		converged = False
		x_change = linalg.norm(x_init)
		m = 0

		while( (converged == False) and (m < max_rep)):
			alpha = 2
			cost_1 = self.cost_function(self.x_opt)
			A = self.calc_A(self.x_opt)

			while(alpha > 0.0000001):
				next_x = linalg.inv(I + alpha*A).dot(I - alpha*A).dot(self.x_opt)
				cost_2 = self.cost_function(next_x)
	
				print alpha, cost_1, cost_2
				if((cost_2 < cost_1) or (abs(cost_1 - cost_2)/abs(cost_1) < 0.0001)):
					x_change = linalg.norm(next_x - self.x_opt)
					[self.x_opt,R] = linalg.qr(next_x)		# QR ensures orthogonality
					self.cost_opt = cost_2
					break
				else:
					alpha = alpha*0.2

			if(x_change < 0.01*linalg.norm(self.x_opt)): converged = True
		if m > 190: print m	
		return self.x_opt	
