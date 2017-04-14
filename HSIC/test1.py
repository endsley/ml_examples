#!/usr/bin/python

from HSIC import *
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
import sklearn.metrics
from sklearn.metrics.cluster import normalized_mutual_info_score




n = 300

#	Linear Data
dat = np.random.rand(n,1)
linear_data = np.hstack((dat,dat)) + 0.04*np.random.randn(n,2)
linear_pc = np.round(pearsonr(linear_data[:,0], linear_data[:,1])[0], 2)
linear_nmi = normalized_mutual_info_score(linear_data[:,0], linear_data[:,1])
print 'linear nmi = ' , linear_nmi
hsic = HSIC(linear_data[:,0], linear_data[:,1], X_kernel='Gaussian', Y_kernel='Gaussian')
hsic_perfect = HSIC(linear_data[:,0], linear_data[:,0], X_kernel='Gaussian', Y_kernel='Gaussian')
linear_hsic = np.round(hsic/hsic_perfect, 2)

#	Sine Data
dat_x = 9.3*np.random.rand(n,1)
dat_y = np.sin(dat_x)
sine_data = np.hstack((dat_x,dat_y)) + 0.1*np.random.randn(n,2)
sine_pc = np.round(pearsonr(sine_data[:,0], sine_data[:,1])[0],2)
sine_nmi = normalized_mutual_info_score(sine_data[:,0], sine_data[:,1])
print 'sine nmi = ' , sine_nmi

sine_hsic = np.round(HSIC(sine_data[:,0], sine_data[:,1], X_kernel='Gaussian', Y_kernel='Gaussian')/HSIC(sine_data[:,0], sine_data[:,0], X_kernel='Gaussian', Y_kernel='Gaussian'), 2)


#	Parabola Data
dat_x = 4*np.random.rand(n,1) - 2
dat_y = 0.05*dat_x*dat_x
para_data = np.hstack((dat_x,dat_y)) + 0.01*np.random.randn(n,2)
para_data[n/2:, 1] = -para_data[n/2:, 1] + 0.4

gamma = 50
para_pc = np.round(pearsonr(para_data[:,0], para_data[:,1])[0],2)
#import pdb; pdb.set_trace()	
para_hsic = np.round(HSIC(para_data[:,0], para_data[:,1], X_kernel='Gaussian', Y_kernel='Gaussian', gamma=gamma)/HSIC(para_data[:,0], para_data[:,0], X_kernel='Gaussian', Y_kernel='Gaussian', gamma=gamma), 2)
print '::', para_hsic

para_nmi = normalized_mutual_info_score(para_data[:,0], para_data[:,1])
print 'para nmi = ' , para_nmi





#	Random uniform Data
unif_data = np.random.rand(n,2)
unif_pc = np.round(pearsonr(unif_data[:,0], unif_data[:,1])[0],2)
unif_hsic = np.round(HSIC(unif_data[:,0], unif_data[:,1], X_kernel='Gaussian', Y_kernel='Gaussian')/HSIC(unif_data[:,0], unif_data[:,0], X_kernel='Gaussian', Y_kernel='Gaussian'), 2)

import pdb; pdb.set_trace()
unif_nmi = normalized_mutual_info_score(unif_data[:,0], unif_data[:,1])
unif_nmi = normalized_mutual_info_score(np.array([1,2,3,4]), np.array([4,2,1,4]))
print 'unif nmi = ' , unif_nmi




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



