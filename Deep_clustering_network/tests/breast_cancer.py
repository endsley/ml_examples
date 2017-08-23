#!/usr/bin/python

import sys
sys.path.append('./lib')
from DCN import *
import numpy as np
from numpy import genfromtxt
import matplotlib 
import matplotlib.pyplot as plt
from Y_2_allocation import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn.metrics
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing


#	load data
data = genfromtxt('datasets/breast-cancer.csv', delimiter=',')
data = preprocessing.scale(data)
label = genfromtxt('datasets/breast-cancer-labels.csv', delimiter=',')

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma = np.median(d_matrix)

num_clusters = 2

Vgamma = 1/(2*sigma*sigma)
allocation = SpectralClustering(num_clusters, gamma=Vgamma).fit_predict(data)
print "NMI : " , normalized_mutual_info_score(allocation, label)



hidden_node_num = 10
dcn = DCN(data, num_clusters, 'breast_cancer', hidden_node_count=hidden_node_num, sigma=sigma, output_d=5)
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
print "NMI : " , normalized_mutual_info_score(allocation, label)







##dcn.initialize_W_to_Gaussian()
#allocation = dcn.run()
##dcn.plot_clustering(allocation)
##allocation = KMeans(2).fit_predict(data)
##Vgamma = 1/(2*sigma*sigma)
##allocation = SpectralClustering(2, gamma=Vgamma).fit_predict(data)
#
#
#if True:	#	output various metrics and info
#	print '\noriginal cost : ' , dcn.original_cost
#	print 'final cost : ' , dcn.final_cost 
#	print "NMI : " , normalized_mutual_info_score(allocation, label)
#	print allocation
#
#
#Y = dcn.NN(dcn.xTor)
#L = dcn.compute_Gaussian_Laplacian(Y, RBF_method='RFF')
#dcn.draw_heatMap(L)
#import pdb; pdb.set_trace()
