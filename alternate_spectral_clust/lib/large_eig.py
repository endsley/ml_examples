#!/usr/bin/python

import numpy as np


def eig_sorted(X):
	D,V = np.linalg.eig(X)	
	idx = D.argsort()[::-1]   
	D = D[idx]
	V = V[:,idx]	

	return [V,D] 


#	Input
#	X is the data matrix itself
#	rank is the expected number of eigen values
#	num_of_random_column is the number of random vectors to project down to
def random_svd(X, rank, num_of_random_column):
	
	if rank + num_of_random_column > X.shape[1]:
		num_of_random_column = X.shape[1] - rank

	random_matrix = np.random.normal(size=(X.shape[1], num_of_random_column))
	omega, r = np.linalg.qr(random_matrix, mode='reduced')
	X_hat = X.dot(X.T)

	Q, R = np.linalg.qr(X_hat.dot(X_hat).dot(X).dot(omega), mode='reduced')
	smaller_matrix = Q.T.dot(X)
	U,S,V = np.linalg.svd(smaller_matrix)
	U = Q.dot(U)

	return U,S,V



#	Note : the sample should be thousands before it start getting accurate

#	X must be positive semidefinite, if not, u must use column sampling on svd
#	sampling_percentage is between 0 to 1
#	note that X3 = [W G21.T; G21 G22]
def nystrom(X, return_rank, sampling_percentage):
	p = sampling_percentage
	num_of_columns = np.floor(p*X.shape[1])
	rp = np.random.permutation(X.shape[1])

	rc = rp[num_of_columns:]	#	residual columns
	rp = rp[0:num_of_columns]	#	random permutation

	X2 = np.hstack((X[:,rp], X[:,rc]))		#	restack horizontally
	X3 = np.vstack((X2[rp,:], X2[rc,:]))	#	restack vertically

	W = X3[0:num_of_columns, 0:num_of_columns]
	G21 = X3[num_of_columns:, 0:num_of_columns]
	G22 = X3[num_of_columns:, num_of_columns:]

	[U,S,V] = random_svd(W, return_rank, return_rank+10)

	ratio = float(X.shape[1])/num_of_columns
	estimated_eig_value = ratio*S[0:return_rank]	
	bottom_estimate = G21.dot(U).dot(np.linalg.inv(np.diag(S)))
	eigVector = np.vstack((U,bottom_estimate))
	eigVector = eigVector[:,0:num_of_columns]

	eigVector = eigVector / np.linalg.norm(eigVector, axis=0)[np.newaxis]
	return [eigVector, estimated_eig_value]


if __name__ == '__main__':

	desired_rank = 5
	example_size = 10000

	X = np.random.normal(size=(example_size, example_size))
	Q,R = np.linalg.qr(X, mode='reduced')
	eigVecs = Q[:,0:desired_rank]
	eigVals = np.diag(np.array(range(desired_rank)) + 1)
	noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise
	
	#for m in range(avg_amount):
	[V,D] = nystrom(M, desired_rank, 0.60)

	print D[0:desired_rank]

