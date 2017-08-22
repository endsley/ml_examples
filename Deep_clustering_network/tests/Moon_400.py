#!/usr/bin/python

from DCN import *
import numpy as np
from numpy import genfromtxt
import matplotlib 
import matplotlib.pyplot as plt
from Y_2_allocation import *
from sklearn.cluster import SpectralClustering

colors = matplotlib.colors.cnames


hidden_node_num = 10

#	load data
data = genfromtxt('datasets/moon_400_2.csv', delimiter=',')

dcn = DCN(data, 2, 'moon_400_2', hidden_node_count=hidden_node_num, sigma=0.3)
dcn.NN = torch.nn.Sequential(
	torch.nn.Linear(dcn.d, dcn.hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(dcn.hidden_d, dcn.hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(dcn.hidden_d, dcn.hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(dcn.hidden_d, dcn.hidden_d, bias=True),
	torch.nn.ReLU(),
	torch.nn.Linear(dcn.hidden_d, dcn.output_d, bias=True),
)

dcn.initialize_W_to_Gaussian()
#dcn.minimize_initial_error()

allocation = dcn.run()


if True:	#	plot the clustering result
	X = data
	plt.figure(1)
	
	plt.subplot(111)
	plt.title('moon')
	idx = np.unique(allocation)
	for mm in idx:
		subgroup = X[allocation == mm]
		plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Alternative Clustering')
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
	plt.show()

#
#if False:		# output info
#	Y = dcn.NN(dcn.xTor)
#	L = dcn.compute_Gaussian_Laplacian(Y, use_RFF=True)
#	dcn.draw_heatMap(L)
#	import pdb; pdb.set_trace()
