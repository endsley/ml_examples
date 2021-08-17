#!/usr/bin/env python

import numpy as np
import sys
import torch
from scipy.special import softmax
from torch.distributions.uniform import Uniform
from sklearn import preprocessing

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=400)
np.set_printoptions(suppress=True)
torch.set_printoptions(edgeitems=3)
#torch.set_printoptions(threshold=10_00)
torch.set_printoptions(linewidth=400)
torch.set_printoptions(sci_mode=False)



def gumbel(pᵢ, τ=0.1, device='cpu'):		#The rows should add up to 1
	'''
		Notes : https://github.com/endsley/math_notebook/blob/master/neural_network/Gumbel_Softmax.pdf

		The gumbel softmax generates samples based on a categorial distribution
		that can be incorporated into a neural network. 
		Given a categorical distribution of {pᑊ pᒾ pᶾ ...}, it will generate one-hot vectors given these probabilities.

		Implement: Make sure that the rows add up to 1
	'''
	if type(pᵢ).__name__ == 'ndarray':	#The rows should add up to 1
		pᵢ = np.atleast_2d(pᵢ)
		logit = np.log(pᵢ)
	
		R = np.random.rand(logit.shape[0], logit.shape[1])
		ε = -np.log(-np.log(R))
		
		noisy_logits = (ε + logit) / τ
		C = softmax(noisy_logits, axis=1)
		return C
	elif type(pᵢ).__name__ == 'Tensor':	#The rows should add up to 1
		pᵢ = pᵢ.to(device, non_blocking=True)
		logit = torch.log(pᵢ)
	
		uniform_dist = Uniform(1e-30, 1.0, )	
		uniform = uniform_dist.rsample(sample_shape=logit.size())
		uniform = uniform.to(device, non_blocking=True )
		ε = -torch.log(-torch.log(uniform))
	
		noisy_logits = (ε + logit) / τ
		C = torch.nn.Softmax(dim=-1)(noisy_logits)
	
		return C

#def Tgumbel(pᵢ, τ=0.1, device='cpu'):	# pytorch implementation of the gumbel-softmax
#	if type(pᵢ).__name__ == 'ndarray':	#The rows should add up to 1
#		pᵢ = torch.from_numpy(pᵢ)
#		pᵢ = pᵢ.to(device, non_blocking=True)
#
#	logit = torch.log(pᵢ)
#
#	uniform_dist = Uniform(1e-30, 1.0, )	
#	uniform = uniform_dist.rsample(sample_shape=logit.size())
#	uniform = uniform.to(device, non_blocking=True )
#	ε = -torch.log(-torch.log(uniform))
#
#	noisy_logits = (ε + logit) / τ
#	C = torch.nn.Softmax(dim=-1)(noisy_logits)
#
#	return C



if __name__ == "__main__": 
	X = np.array([[0.4405, 0.4045, 0.0754, 0.0796],			# The rows should add up to 1
					[0.2287, 0.2234, 0.2676, 0.2802],
					[0.2518, 0.2524, 0.1696, 0.3262],
					[0.2495, 0.1744, 0.2126, 0.3635],
					[0.1979, 0.31  , 0.2165, 0.2755],
					[0.2003, 0.2329, 0.2982, 0.2686]])
	
	# numpy implementation
	C = np.zeros((6,4))
	for n in range(10000):
		C += np.round(gumbel(X))
	
	C = C/10000
	
	# torch implementation
	Ct = torch.zeros(6,4)
	for n in range(10000):
		Ct += torch.round(gumbel(torch.from_numpy(X)))
	
	Ct = Ct.numpy()/10000

	print('True Probability')	
	print(X,'\n')

	print('Gumbel generated Probability from sampling')
	print(C,'\n')

	print('Gumbel generated Probability from sampling via pytorch')
	print(Ct,'\n')
