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
data = genfromtxt('datasets/contraceptive_data.csv', delimiter=',')
data = preprocessing.scale(data)
label = genfromtxt('datasets/contraceptive_label.csv', delimiter=',')

d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
sigma = np.median(d_matrix)

num_clusters = 3
Vgamma = 1/(2*sigma*sigma)
allocation = SpectralClustering(num_clusters, gamma=Vgamma).fit_predict(data)
print "Spectral Cluster NMI : " , normalized_mutual_info_score(allocation, label)


allocation = KMeans(num_clusters).fit_predict(data)
print "Kmeans NMI : " , normalized_mutual_info_score(allocation, label)




for outd in range(1, 20):
	for hidden in range(2,30):
		dcn = DCN(data, num_clusters, 'contraception', hidden_node_count=hidden, sigma=sigma, output_d=outd)
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
		print outd, ' : ' , hidden , "  NMI : " , normalized_mutual_info_score(allocation, label)



