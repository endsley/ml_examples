#!/usr/bin/env python
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import numpy as np
from hsic import *
import torch

torch.set_printoptions(edgeitems=3)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=2)
torch.set_printoptions(linewidth=400)
np.set_printoptions(edgeitems=3)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=400)


def torch_hsic(X, Yₒ, with_normalized_H=True):
	X = torch.tensor(X)
	Yₒ = torch.tensor(Yₒ)

	σ = (torch.nn.functional.pdist(X)).median()
	γ = 1.0 / (2 * σ * σ)
	K = torch.exp(γ * -torch.norm(X.unsqueeze(1) - X, dim=2) ** 2)

	Kᵧ = Yₒ@Yₒ.T
	HKᵪ = K - torch.mean(K, dim=0) 
	HKᵧ = Kᵧ - torch.mean(Kᵧ, dim=0)

	Hᵪᵧ = torch.sum(HKᵪ.T*HKᵧ)

	if with_normalized_H:
		Hᵪ = torch.norm(HKᵪ)						# equivalent to 	np.sqrt(np.sum(KᵪH*KᵪH))
		Hᵧ = torch.norm(HKᵧ) 						# equivalent to 	np.sqrt(np.sum(KᵧH*KᵧH))
		hsic = Hᵪᵧ/( Hᵪ * Hᵧ )
		return hsic
	else:
		return Hᵪᵧ

def numpy_hsic(X, Yₒ, with_normalized_H=True):	# not optimized
	σ = (torch.nn.functional.pdist(torch.tensor(X))).median()
	n = X.shape[0]

	γ = 1.0 / (2 * σ * σ)
	H = np.eye(n) - (1/n)*np.ones((n,n))
	Kᵪ = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ.item())
	Kᵧ = Yₒ@Yₒ.T
	Hᵪᵧ = np.trace(H@Kᵪ@H@Kᵧ)

	if with_normalized_H:
		HKᵪ = Kᵪ - np.mean(Kᵪ, axis=0) 
		HKᵧ = Kᵧ - np.mean(Kᵧ, axis=0)

		Hᵪ = np.linalg.norm(HKᵪ)						# equivalent to 	np.sqrt(np.sum(KᵪH*KᵪH))
		Hᵧ = np.linalg.norm(HKᵧ) 						# equivalent to 	np.sqrt(np.sum(KᵧH*KᵧH))
		H = Hᵪᵧ/( Hᵪ * Hᵧ )
		return H
	else:
		return Hᵪᵧ


if __name__ == '__main__':
	X = np.vstack((np.random.randn(4,2), np.random.randn(4,2) + 5))
	Y = np.hstack((np.zeros(4), np.ones(4)))
	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.atleast_2d(Y).T)
		
	torch_H = torch_hsic(X, Yₒ)
	numpy_H = numpy_hsic(X, Yₒ)

	print('Numpy HSIC : %.2f, Torch HSIC : %.2f'%(numpy_H, torch_H))
	import pdb; pdb.set_trace()

