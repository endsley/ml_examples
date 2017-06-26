#!/usr/bin/python

from HSIC_rbf import *
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
import sklearn.metrics





#	Linear Data
n = 300
dat = np.random.rand(n,1)
linear_data = np.hstack((dat,dat)) + 0.04*np.random.randn(n,2)
linear_pc = np.round(pearsonr(linear_data[:,0], linear_data[:,1])[0], 2)
linear_hsic = np.round(HSIC_rbf(linear_data[:,0], linear_data[:,1], 1)/HSIC_rbf(linear_data[:,0], linear_data[:,0], 1), 2)


#	Sine Data
dat_x = 9.3*np.random.rand(n,1)
dat_y = np.sin(dat_x)
sine_data = np.hstack((dat_x,dat_y)) + 0.1*np.random.randn(n,2)
sine_pc = np.round(pearsonr(sine_data[:,0], sine_data[:,1])[0],2)
sine_hsic = np.round(HSIC_rbf(sine_data[:,0], sine_data[:,1], 1)/HSIC_rbf(sine_data[:,0], sine_data[:,0], 1), 2)


#	Parabola Data
dat_x = 4*np.random.rand(n,1) - 2
dat_y = 0.05*dat_x*dat_x
para_data = np.hstack((dat_x,dat_y)) + 0.01*np.random.randn(n,2)
para_data[n/2:, 1] = -para_data[n/2:, 1] + 0.4

#d_matrix = sklearn.metrics.pairwise.pairwise_distances(para_data, Y=None, metric='euclidean')
#sigma = np.median(d_matrix)
#print sigma

sigma = 0.1
para_pc = np.round(pearsonr(para_data[:,0], para_data[:,1])[0],2)
para_hsic = np.round(HSIC_rbf(para_data[:,0], para_data[:,1], sigma)/HSIC_rbf(para_data[:,0], para_data[:,0], sigma), 2)
#print '::', para_hsic

#	Random uniform Data
unif_data = np.random.rand(n,2)
unif_pc = np.round(pearsonr(unif_data[:,0], unif_data[:,1])[0],2)
unif_hsic = np.round(HSIC_rbf(unif_data[:,0], unif_data[:,1], 1)/HSIC_rbf(unif_data[:,0], unif_data[:,0], 1), 2)



plt.subplot(141)
plt.plot(linear_data[:,0], linear_data[:,1], 'bx')
#plt.ylabel('Y')
#plt.xlabel('X')
plt.title('Corr : ' + str(linear_pc) + ' , HSIC : ' + str(linear_hsic))

plt.subplot(142)
plt.plot(sine_data[:,0], sine_data[:,1], 'bx')
#plt.ylabel('Y')
#plt.xlabel('X')
plt.title('Corr : ' + str(sine_pc) + ' , HSIC : ' + str(sine_hsic))


plt.subplot(143)
plt.plot(para_data[:,0], para_data[:,1], 'bx')
#plt.ylabel('Y')
#plt.xlabel('X')
plt.title('Corr : ' + str(para_pc) + ' , HSIC : ' + str(para_hsic))


plt.subplot(144)
plt.plot(unif_data[:,0], unif_data[:,1], 'bx')
#plt.ylabel('Y')
#plt.xlabel('X')
plt.title('Corr : ' + str(unif_pc) + ' , HSIC : ' + str(unif_hsic))



plt.show()



