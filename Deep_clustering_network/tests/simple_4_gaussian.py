#!/usr/bin/python

import sys
sys.path.append('./lib')
from DCN import *
import numpy as np
from numpy import genfromtxt
import matplotlib 
import matplotlib.pyplot as plt
from Y_2_allocation import *

colors = matplotlib.colors.cnames



#	load data
data = genfromtxt('datasets/data_4.csv', delimiter=',')
Y_original = genfromtxt('datasets/data_4_Y_original.csv', delimiter=',')
U_original = genfromtxt('datasets/data_4_U_original.csv', delimiter=',')


dcn = DCN(data,4, 'model_20_neurons')
dcn.hidden_d = dcn.d + 2000
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
	torch.nn.Sigmoid(),
)

#dcn.initialize_W()

allocation = dcn.run()
import pdb; pdb.set_trace()
#if True:	#	plot the clustering result
#	X = data
#	plt.figure(1)
#	
#	plt.subplot(111)
#	plt.title('data_4.csv original plot')
#	idx = np.unique(allocation)
#	for mm in idx:
#		subgroup = X[allocation == mm]
#		plt.plot(subgroup[:,0], subgroup[:,1], color=colors.keys()[int(mm)] , marker='o', linestyle='None')
#	plt.xlabel('Feature 1')
#	plt.ylabel('Feature 2')
#	plt.title('Alternative Clustering')
#	
#	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#	plt.show()
#
#
