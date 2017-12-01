#!/usr/bin/env python

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import numpy as np

#np.set_printoptions(precision=1)
#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(linewidth=300)
#np.set_printoptions(suppress=True)


x = np.loadtxt('../dataset/data_4.csv', delimiter=',', dtype=np.float32)
y = np.loadtxt('../dataset/data_4_label.csv', delimiter=',', dtype=np.int32)


#clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 50, 50), random_state=1) #lbfgs
clf = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 50, 50), random_state=1, learning_rate='constant', batch_size=40 ) #lbfgs, invscaling


clf.fit(x, y)  
#print clf.score(x, y)
#print clf.predict_proba(x)
print clf.predict(x)
#print clf.predict([[-0.1,-0.1]])
