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

np.set_printoptions(threshold=np.nan)
mat_contents = sio.loadmat('Euclid.mat')
num_of_rows = len(mat_contents['Euclid_matrix'])
similarity_matrix = np.empty((0,num_of_rows), float)


def Get_row_similarity(idx):
	print idx
	row = mat_contents['Euclid_matrix'][:,idx]	
	min_covariance = np.min(np.abs(np.diff(row)))/100;

	bic_vals = np.array([])
	prediction_table = {}
	bic_table = {}
	for m in range(2,10):
		gmm = GMM(n_components=m, covariance_type='diag', init_params='wc', min_covar=min_covariance) 
		gmm.fit(row)
		cluster_mean = gmm.means_
	
		sorted_indices = np.argsort(cluster_mean, axis=None) # get index of 2 smallest mean value
		bic_rating = round(gmm.bic(row),2)
		prediction_results = gmm.predict(row)
		
		#print prediction_results
		#print cluster_mean
		#print bic_rating

		if cluster_mean.shape[0] != np.unique(prediction_results).shape[0]: 
			break

			#print 'shape : ' , np.unique(prediction_results).shape[0]
			#if np.unique(prediction_results).shape[0] == 1: continue
			#if np.unique(prediction_results).shape[0] > 1:
			#	print 'Mis-match break : ' , cluster_mean.shape[0], np.unique(prediction_results).shape[0]
			#	break


		prediction_table[m] = [prediction_results, sorted_indices[0:2], cluster_mean]
		bic_vals = np.append(bic_vals, bic_rating)
		bic_table[bic_rating] = m
	
	min_bic_vals = np.min(bic_vals)
	assignment = prediction_table[bic_table[min_bic_vals]][0]
	two_smallest_indices = prediction_table[bic_table[min_bic_vals]][1]
	all_means = prediction_table[bic_table[min_bic_vals]][2]
	unique_assignments = np.unique(assignment)

	cluster_1 = row[assignment == two_smallest_indices[0]]
	cluster_2 = row[assignment == two_smallest_indices[1]]
	
	#print assignment
	#print two_smallest_indices[0]
	#print two_smallest_indices[1]
	
	#print cluster_1.shape
	#print cluster_2.shape
	
	X = np.expand_dims(np.transpose(np.append(cluster_1, cluster_2)),0)
	Y = np.transpose(np.append(np.ones(cluster_1.shape[0]), np.zeros(cluster_2.shape[0])))
	
	lower_range = np.min(row)
	upper_range = np.max(row)
	increment = (upper_range - lower_range)/20.0

	logistic_range = np.arange(lower_range, upper_range, increment)
	logistic_range = np.expand_dims(logistic_range,1)

	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(np.transpose(X), Y)
	prob_matrix = logreg.predict_proba(np.expand_dims(row,1))

	sigmoid = logreg.predict_proba(logistic_range)
	sigmoid = 7*sigmoid[:,1]

	row_similarity = np.expand_dims(prob_matrix[:,1],0)


	f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
#	print cluster_1
#	print cluster_2
#	print 'lower : ' , lower_range
#	print 'upper : ' , upper_range
#	print 'inc : ' , increment
#	print 'assignment " ' , assignment
#	print 'Cluster 1 mean : ', np.mean(cluster_1)
#	print 'Cluster 2 mean : ', np.mean(cluster_2)
#	print two_smallest_indices
#	print bic_vals
#	all_means = np.transpose(np.sort(all_means,0))
#	print 'All means : ', all_means
	ax1.hist(row,80)
	ax1.plot(logistic_range, sigmoid)
	#ax2.plot(row_similarity)
	ax2.plot(range(2,bic_vals.shape[0] + 2), bic_vals)
	#plt.show()
	plt.savefig('all_imgs/output_' + str(idx) + '.png')


	return row_similarity
	



for i in range(num_of_rows):
	row_similarity = Get_row_similarity(i)
	print similarity_matrix.shape

	similarity_matrix = np.vstack((similarity_matrix,row_similarity))

sio.savemat('similarity_matrix.mat', {'similarity_matrix':similarity_matrix})


#for i in range(20):
#	row_similarity = Get_row_similarity(i)

#row_similarity = Get_row_similarity_kmeans(56)
#print row_similarity
