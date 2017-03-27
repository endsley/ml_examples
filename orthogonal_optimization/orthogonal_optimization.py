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
		max_rep = 3
		converged = False
		m = 0

		while( (converged == False) and (m < max_rep)):
			alpha = 2
			cost_1 = self.cost_function(self.x_opt)
			A = self.calc_A(self.x_opt)

			while(alpha > 0.01):
				next_x = linalg.inv(I + alpha*A).dot(I - alpha*A).dot(self.x_opt)
				cost_2 = self.cost_function(next_x)
	
				if(cost_2 < cost_1):
					x_change = linalg.norm(next_x - self.x_opt)
					self.x_opt = next_x
					self.cost_opt = cost_2
					break
				else:
					alpha = alpha*0.2

			if(x_change < 0.01*linalg.norm(self.x_opt)): converged = True
			
		return self.x_opt	
