#!/usr/bin/python

from DCN import *
import numpy as np
from numpy import genfromtxt
import matplotlib 
import matplotlib.pyplot as plt
from Y_2_allocation import *
from sklearn import preprocessing



hidden_node_num = 10

#	load data
data = genfromtxt('datasets/Two_Gauss_400_2.csv', delimiter=',')
data = preprocessing.scale(data)		# center and scaled

dcn = DCN(data, 2, 'Two_Gauss_400_2', hidden_node_count=hidden_node_num, sigma=1, output_d=2)
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
allocation = dcn.run()
dcn.plot_clustering(allocation=allocation)
print allocation

#if False:		# output info
#	Y = dcn.NN(dcn.xTor)
#	L = dcn.compute_Gaussian_Laplacian(Y, use_RFF=True)
#	dcn.draw_heatMap(L)
#	import pdb; pdb.set_trace()
