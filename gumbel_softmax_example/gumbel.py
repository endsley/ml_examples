#!/usr/bin/env python

import numpy as np
import sys
import torch
from scipy.special import softmax
from torch.distributions.uniform import Uniform


np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=400)
np.set_printoptions(suppress=True)
torch.set_printoptions(edgeitems=3)
#torch.set_printoptions(threshold=10_00)
torch.set_printoptions(linewidth=400)
torch.set_printoptions(sci_mode=False)

def gumbel(logit):
	τ = 0.1
	logit = np.atleast_2d(logit)
	R = np.random.rand(logit.shape[0], logit.shape[1])
	ε = -np.log(-np.log(R))
	
	noisy_logits = (ε + logit) / τ
	C = softmax(noisy_logits, axis=1)
	return C


def gumbel_torch(logit, τ=0.1, device='cpu'):
	if type(logit) == type(np.array([])):
		logit = torch.from_numpy(logit)
		logit = logit.to(device, non_blocking=True)

	#uniform_dist = Uniform(1e-30, 1.0, )	
	#uniform = uniform_dist.rsample(sample_shape=logit.size())
	#uniform = uniform.to(device, non_blocking=True )

	uniform = torch.tensor([[0.0395, 0.5627, 0.6333, 0.9879],
							[0.9708, 0.5521, 0.9495, 0.7506],
							[0.5467, 0.4756, 0.7682, 0.4536],
							[0.2372, 0.6185, 0.1950, 0.8943],
							[0.3845, 0.2134, 0.5381, 0.2988],
							[0.1815, 0.2892, 0.7473, 0.7980]])

	ε = -torch.log(-torch.log(uniform))
	noisy_logits = (ε + logit) / τ
	C = torch.nn.Softmax(dim=-1)(noisy_logits)
	print(C)
	import pdb; pdb.set_trace()
	return C



#X = np.random.rand(8,3)
X = torch.tensor([[0.6120, 0.7751, 0.4464, 0.7116],
					[0.7559, 0.7382, 0.8844, 0.9260],
					[0.8406, 0.8425, 0.5663, 1.0891],
					[0.6051, 0.4229, 0.5154, 0.8814],
					[0.5818, 0.9113, 0.6363, 0.8099],
					[0.5310, 0.6175, 0.7904, 0.7120]])
C = gumbel(X)
Ct = gumbel_torch(X)

print(X,'\n')
print(C)
print(Ct)
