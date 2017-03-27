#!/usr/bin/python

from orthogonal_optimization import *
from numpy import *

def cost_function(x):
	return sum(x)	

def gradient_function(x):
	return array([[1],[1]])

x_init = array([[1],[0]])
OO = orthogonal_optimization(cost_function, gradient_function)
x_opt = OO.run(x_init)
print x_opt
print OO.cost_opt


#min_cost = 10
#for m in range(10000000):
#	x = random.randn(2,1)
#	x = x/linalg.norm(x)
#	cost = cost_function(x)
#	if cost < min_cost: 
#		min_cost = cost
#		print x, ' : ' , min_cost
