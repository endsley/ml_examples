#!/usr/bin/python

import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GMM
from sklearn import linear_model, datasets
#from tempfile import TemporaryFile



np.set_printoptions(threshold=np.nan)
mat_contents = sio.loadmat('Euclid.mat')
num_of_rows = len(mat_contents['Euclid_matrix'])
similarity_matrix = np.empty((0,num_of_rows), float)


def Get_row_similarity(idx):
	row = mat_contents['Euclid_matrix'][:,idx]	
	#row = np.square(row)

	min_covariance = np.min(np.abs(np.diff(row)))/100;

	bic_vals = np.array([])
	prediction_table = {}
	bic_table = {}
	for m in range(2,11):
		gmm = GMM(n_components=m, covariance_type='diag', init_params='wc', min_covar=min_covariance) 
		gmm.fit(row)
		cluster_mean = gmm.means_
	
		sorted_indices = np.argsort(cluster_mean, axis=None) # get index of 2 smallest mean value
		bic_rating = round(gmm.bic(row),2)
		prediction_results = gmm.predict(row)
		
		if cluster_mean.shape[0] != np.unique(prediction_results).shape[0]: 
			continue

			#print 'shape : ' , np.unique(prediction_results).shape[0]
			#if np.unique(prediction_results).shape[0] == 1: continue
			#if np.unique(prediction_results).shape[0] > 1:
			#	print 'Mis-match break : ' , cluster_mean.shape[0], np.unique(prediction_results).shape[0]
			#	break


		prediction_table[m] = [prediction_results, sorted_indices, cluster_mean]
		bic_vals = np.append(bic_vals, bic_rating)
		bic_table[bic_rating] = m
	

	min_bic_vals = np.min(bic_vals)
	assignment = prediction_table[bic_table[min_bic_vals]][0]
	two_smallest_indices = prediction_table[bic_table[min_bic_vals]][1]
	all_means = prediction_table[bic_table[min_bic_vals]][2]
	unique_assignments = np.unique(assignment)

	cluster_1 = row[assignment == two_smallest_indices[0]]
	cluster_2 = row[assignment == two_smallest_indices[1]]
	if len(two_smallest_indices) > 2:
		cluster_3 = row[assignment == two_smallest_indices[2]]

	#	Make sure cluster_1 is not too small
	if len(two_smallest_indices) > 2:
		print str(idx) + ' cluster merged due small cluster 1'
		if len(cluster_1) < 10:
			cluster_1 = np.append(cluster_1, cluster_2)
			cluster_2 = cluster_3

	#	Make sure cluster_1 and 2 are not too close
	mean_1 = np.mean(cluster_1)
	mean_2 = np.mean(cluster_2)
	std_1 = np.std(cluster_1)

	#print 'mean 1 : ' , mean_1
	#print 'mean 2 : ' , mean_2
	#print 'top : ' , mean_1 + std_1/100.0
	#print 'bottom : ' , mean_1 - std_1/100.0
	if((mean_2 < mean_1 + std_1) and (mean_2 > mean_1 - std_1)):
		print str(idx) + ' cluster merged due to mean proximity'
		total_cluster = np.sort(np.append(cluster_1, cluster_2))
		cluster_1 = total_cluster[0:len(total_cluster) - 4]
		cluster_2 = total_cluster[-int(np.floor(len(total_cluster)/3.0)):len(total_cluster)]

	cluster_1 = np.sort(cluster_1)
	cluster_2 = np.sort(cluster_2)

	cluster_len = len(cluster_1)	
	first_half_len = int(np.ceil(cluster_len/2.0))
	cluster_2_half_len = int(np.floor(len(cluster_2)/2))
	second_half_len = cluster_len - first_half_len + len(cluster_2) 
	first_half = cluster_1[0: first_half_len]
	two_half = cluster_2[0: cluster_2_half_len]
	second_rest = cluster_1[first_half_len:cluster_len]

	X = np.expand_dims(np.append(np.append(np.append(cluster_1, two_half), second_rest), cluster_2),0)
	Y = np.transpose(np.append(np.ones(cluster_len + cluster_2_half_len), np.zeros(second_half_len)))

#	print 'Cluster 1 mean : ', np.mean(cluster_1)
#	print 'Cluster 2 mean : ', np.mean(cluster_2)
#	print 'Cluster 1 :', cluster_1
#	print '\n'
#	print 'Cluster 2 :', cluster_2
#	print int(cluster_len - 10)
#	print 'cluster len : ' , cluster_1.shape
#	print '\n'
#	print 'first_half_len : ' , first_half_len
#	print '\n'
#	print second_rest
#	print '2nd rest : ' , second_rest.shape
#	print '\n'
#	print 'Cluster 2: ', cluster_2
#	print '\n'
#	print 'Cluster 2 half: ', two_half
#	print X.shape
#	print X
#	print Y.shape
#	print Y


	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(np.transpose(X), Y)
	prob_matrix = logreg.predict_proba(np.expand_dims(row,1))

	row_similarity = np.expand_dims(prob_matrix[:,1],0)
	#print prob_matrix

	lower_range = np.min(row)
	upper_range = np.max(row)
	increment = (upper_range - lower_range)/20.0

	logistic_range = np.arange(lower_range, upper_range, increment)
	logistic_range = np.expand_dims(logistic_range,1)
	sigmoid = logreg.predict_proba(logistic_range)
	sigmoid = sigmoid[:,1]


	

	all_means = np.transpose(np.sort(all_means,0))
	print str(idx) + ' all means : ', all_means
	f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
	ax1.set_title('Histogram and the likelihood drop')
	histVals = ax1.hist(row,80)
	max_hist = np.max(histVals[0])
	ax1.plot(logistic_range, max_hist*sigmoid)
	ax1.plot([np.median(row), np.median(row)], [0,max_hist])
	ax1.text(increment, max_hist-2, all_means)
	#ax2.plot(row_similarity)
	ax2.set_title('BIC model selection')
	ax2.plot(range(2,bic_vals.shape[0] + 2), bic_vals)
	#plt.show()
	plt.savefig('histogram_graphs/output_' + str(idx) + '.png')
	plt.close(f)

	return row_similarity


for i in range(num_of_rows):
	row_similarity = Get_row_similarity(i)
	similarity_matrix = np.vstack((similarity_matrix,row_similarity))

sio.savemat('similarity_matrix.mat', {'similarity_matrix':similarity_matrix})


#for i in range(20):
#row_similarity = Get_row_similarity(42) #38

#print row_similarity
