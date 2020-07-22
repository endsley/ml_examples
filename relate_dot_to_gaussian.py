#!/usr/bin/env python

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket
import sklearn.metrics
from scipy.optimize import minimize
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.random import sample_without_replacement

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


#	Experiment
#	If we were to randomly generate 2 unit vectors 
#	As the dimension of the vector increase, 
#		how does the average inner product and average distance behave?
#	Conclusion: it seems the they become orthogonal. 


class dot_to_gaussian():
	def __init__(self):
		num_sample = 2000

		print('{:10}{:15}{:10}{:10}{:10}{:10}'.format('Dim', '#samples','Dot μ', 'Dot σ', 'D μ', 'D σ'))
		for dim in np.arange(10,200,10):
			dis_list = []
			dp_list = []
	
			for p in range(num_sample):
				while True:
					self.x = np.random.randn(dim,1)
					self.y = np.random.randn(dim,1)
		
					self.x = self.x/np.linalg.norm(self.x)
					self.y = self.y/np.linalg.norm(self.y)
		
					dp = self.x.T.dot(self.y)
					if dp > 0:
						break;
		
				d = np.linalg.norm(self.x-self.y)	
				dis_list.append(d)
				dp_list.append(dp)
	
			dot_μ = (np.mean(dp_list))
			dot_σ = (np.std(dp_list))
			D_μ = (np.mean(dis_list))
			D_σ = (np.std(dis_list))
	
	
			print('{:<10d}{:<15d}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}'.format(dim, num_sample, dot_μ, dot_σ, D_μ, D_σ))


		for dim in np.arange(1000,20000,1000):
			dis_list = []
			dp_list = []
	
			for p in range(num_sample):
				while True:
					self.x = np.random.randn(dim,1)
					self.y = np.random.randn(dim,1)
		
					self.x = self.x/np.linalg.norm(self.x)
					self.y = self.y/np.linalg.norm(self.y)
		
					dp = self.x.T.dot(self.y)
					if dp > 0:
						break;
		
				d = np.linalg.norm(self.x-self.y)	
				dis_list.append(d)
				dp_list.append(dp)
	
			dot_μ = (np.mean(dp_list))
			dot_σ = (np.std(dp_list))
			D_μ = (np.mean(dis_list))
			D_σ = (np.std(dis_list))
	
	
			print('{:<10d}{:<15d}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(dim, num_sample, dot_μ, dot_σ, D_μ, D_σ))









			
k = dot_to_gaussian()
