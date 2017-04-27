#!/usr/bin/python

import sys
import Nice4Py
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def doHeatmap(X, labels, path):
    Y = linkage(X, method='ward')
    fig = plt.figure(figsize=(10,10))
    ax_heat = fig.add_axes([0, 0, 1, 0.7])
    ax_dendro = fig.add_axes([0, 0.75, 1, 0.25])
    Z = dendrogram(Y, ax=ax_dendro, labels=labels, leaf_font_size=5, leaf_rotation=80)
    ax_dendro.yaxis.set_visible(False)
    for pos in ['left', 'right', 'bottom', 'top']:
        ax_dendro.spines[pos].set_visible(False)
    index = Z['leaves']
    X = X[:, index]
    X = X[index, :]
    ax_heat.matshow(X, aspect='auto', cmap=plt.cm.Reds)
    ax_heat.axis('off')
    fig.savefig(path)

data_type = np.float32
data = np.genfromtxt('moon.csv', delimiter=',', dtype=data_type)
num_samples = data.shape[0]
num_features = data.shape[1]
num_clusters = 2
output = np.empty((num_samples, 1), dtype=data_type)
kernel_matrix = np.empty((num_samples, num_samples), dtype=data_type)


#kdac = Nice4Py.KDAC(sys.argv[1])
kdac = Nice4Py.KDAC('gpu')

params = {'c':num_clusters, 'q':num_clusters, 'kernel':'Gaussian', 'lambda':1.0, 'sigma':0.3, 'verbose':1.0}
kdac.SetupParams(params)
print 'First Fit to Generate Y Matrix'
kdac.Fit(data, num_samples, num_features)
print 'Second Fit'
kdac.Fit()
kdac.Predict(output, num_samples, 1)
kdac.GetK(kernel_matrix, num_samples)
print 'Sigma = 0.3, First Alternative: '
print output.T
print '============================================='
print kernel_matrix
print '============================================='
labels = [str(x) for x in np.arange(kernel_matrix.shape[0])]
doHeatmap(kernel_matrix, labels, "Heatmap.pdf")



kdac.DiscardLastRun()
params['sigma'] = 1.0
kdac.SetupParams(params)
print 'Sigma = 1.0, last result discarded'
kdac.Fit()
kdac.Predict(output, num_samples, 1)
print output.T
print '============================================='

kdac.DiscardLastRun()
params['sigma'] = 0.3
kdac.SetupParams(params)
print 'Sigma = 0.3, last result discarded'
kdac.Fit()
kdac.Predict(output, num_samples, 1)
print output.T

#kdac.Fit(data, num_samples, num_features, y1, num_samples, num_clusters)
#np.savetxt('alternative.csv', output+1, delimiter=',')

# If you want to get matrix U
# the syntax is as below
# kdac.GetU(output, <row number of U>, <row number of V>)
# Getting Matrix W uses same syntax

# Another Getters GetQ() GetD() GetN(), No argument needed
#q = kdac.GetQ();
#d = kdac.GetD();
#n = kdac.GetN();
#print q
#print d
#print n
