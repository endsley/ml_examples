#!/usr/bin/env python

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

dname = 'wine'


def use_svm(X,Y,k='rbf', K=None):	
	svm_object = svm.SVC(kernel=k)

	if K is None:
		svm_object.fit(X, Y)
		out_allocation = svm_object.predict(X)
	else:
		svm_object.fit(K, Y)
		out_allocation = svm_object.predict(K)

	#nmi = normalized_mutual_info_score(out_allocation, Y)
	acc = accuracy_score(out_allocation, Y)

	return [out_allocation, acc, svm_object]


for i in range(1, 11):
	file_name = './data/' + dname + '/' + dname + '_' + str(i) + '.csv'
	label = './data/' + dname + '/' + dname + '_' + str(i) + '_label.csv'

	X = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
	X = preprocessing.scale(X)
	Y = np.loadtxt(label, delimiter=',', dtype=np.int32)


	pca = PCA(n_components=0.8)
	X_pca = pca.fit_transform(X)

	[allocation, acc, svm_object] = use_svm(X,Y,k='rbf')
	print('acc : %.3f'%acc)
	import pdb; pdb.set_trace()

	#print(Y.shape)
	#print(i)
