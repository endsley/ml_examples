#!/usr/bin/env python

import numpy as np
import sklearn.metrics
import sklearn.metrics.pairwise
from scipy.stats.stats import pearsonr  
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
import ppscore as pps

from Dependency.HSIC import HSIC as ℍ

n = 300

#	Perfect Linear Data
dat = np.random.rand(n,1)
plinear_data = np.hstack((dat,dat)) + 1
df = pd.DataFrame(data=plinear_data, columns=["x", "y"])

enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
XP_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(plinear_data[:,0]).T))
enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
YP_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(plinear_data[:,1]).T))

plinear_pc = np.round(pearsonr(plinear_data[:,0], plinear_data[:,1])[0], 2)
plinear_nmi = np.round(normalized_mutual_info_score(XP_data_nmi, YP_data_nmi),2)
plinear_hsic = np.round(ℍ(plinear_data[:,0], plinear_data[:,1], sigma_type='opt'),2)	
plinear_pps = np.round(pps.score(df, "x", "y")['ppscore'],2)

print('Linear Relationship:')
print('\tCorrelation : ', plinear_pc)
print('\tNMI : ', plinear_nmi)
print('\tpps : ', plinear_pps)
print('\tHSIC : ', plinear_hsic)


#	Linear Data
dat = np.random.rand(n,1)
linear_data = np.hstack((dat,dat)) + 0.04*np.random.randn(n,2)
df = pd.DataFrame(data=linear_data, columns=["x", "y"])

enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
XL_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(linear_data[:,0]).T))
enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
YL_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(linear_data[:,1]).T))

linear_pc = np.round(pearsonr(linear_data[:,0], linear_data[:,1])[0], 2)
linear_nmi = np.round(normalized_mutual_info_score(XL_data_nmi, YL_data_nmi),2)
linear_hsic = np.round(ℍ(linear_data[:,0], linear_data[:,1], sigma_type='opt'),2)	
linear_pps = np.round(pps.score(df, "x", "y")['ppscore'],2)

print('Linear Relationship:')
print('\tCorrelation : ', linear_pc)
print('\tNMI : ', linear_nmi)
print('\tpps : ', linear_pps)
print('\tHSIC : ', linear_hsic)

#	Sine Data
dat_x = 9.3*np.random.rand(n,1)
dat_y = np.sin(dat_x)
sine_data = np.hstack((dat_x,dat_y)) + 0.06*np.random.randn(n,2)
df = pd.DataFrame(data=sine_data, columns=["x", "y"])
sine_pc = np.round(pearsonr(sine_data[:,0], sine_data[:,1])[0],2)

enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Xsine_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(sine_data[:,0]).T))
enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Ysine_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(sine_data[:,1]).T))

sine_nmi = np.round(normalized_mutual_info_score(Xsine_data_nmi, Ysine_data_nmi),2)
sine_hsic = np.round(ℍ(sine_data[:,0], sine_data[:,1], sigma_type='opt'),2)
sine_pps = np.round(pps.score(df, "x", "y")['ppscore'],2)

print('Sine Relationship:')
print('\tCorrelation : ', sine_pc)
print('\tNMI : ', sine_nmi)
print('\tpps : ', sine_pps)
print('\tHSIC : ', sine_hsic)


#	Parabola Data
dat_x = 4*np.random.rand(n,1) - 2
dat_y = 0.05*dat_x*dat_x
para_data = np.hstack((dat_x,dat_y)) + 0.01*np.random.randn(n,2)
df = pd.DataFrame(data=para_data, columns=["x", "y"])

enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Xp_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(para_data[:,0]).T))
enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Yp_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(para_data[:,1]).T))

para_pc = np.round(pearsonr(para_data[:,0], para_data[:,1])[0],2)
para_nmi = np.round(normalized_mutual_info_score(Xp_data_nmi, Yp_data_nmi),2)
para_hsic = np.round(ℍ(para_data[:,0], para_data[:,1], sigma_type='opt'),2)
para_pps = np.round(pps.score(df, "x", "y")['ppscore'],2)

print('Parabola Relationship:')
print('\tCorrelation : ', para_pc)
print('\tNMI : ', para_nmi)
print('\tpps : ', para_pps)
print('\tHSIC : ', para_hsic)

#	Random uniform Data
unif_data = np.random.rand(n,2)
df = pd.DataFrame(data=unif_data, columns=["x", "y"])

enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Xr_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(unif_data[:,0]).T))
enc = KBinsDiscretizer(n_bins=10, encode='ordinal')
Yr_data_nmi = np.squeeze(enc.fit_transform(np.atleast_2d(unif_data[:,1]).T))


unif_pc = np.round(pearsonr(unif_data[:,0], unif_data[:,1])[0],2)
unif_hsic = np.round(ℍ(unif_data[:,0], unif_data[:,1], sigma_type='opt'),2)
unif_nmi = np.round(normalized_mutual_info_score(Xr_data_nmi, Yr_data_nmi),2)
unif_pps = np.round(pps.score(df, "x", "y")['ppscore'],2)

print('Random Relationship:')
print('\tCorrelation : ', unif_pc)
print('\tNMI : ', unif_nmi)
print('\tpps : ', unif_pps)
print('\tHSIC : ', unif_hsic)


plt.figure(figsize=(13,3))

plt.subplot(151)
plt.plot(plinear_data[:,0], plinear_data[:,1], 'bx')
plt.title('$\\rho$ : ' + str(plinear_pc) + ' , HSIC : ' + str(plinear_hsic) + '\npps : ' + str(plinear_pps) + ' , nmi : ' + str(plinear_nmi))

plt.subplot(152)
plt.plot(linear_data[:,0], linear_data[:,1], 'bx')
plt.title('$\\rho$ : ' + str(linear_pc) + ' , HSIC : ' + str(linear_hsic) + '\npps : ' + str(linear_pps) + ' , nmi : ' + str(linear_nmi))

plt.subplot(153)
plt.plot(sine_data[:,0], sine_data[:,1], 'bx')
plt.title('$\\rho$ : ' + str(sine_pc) + ' , HSIC : ' + str(sine_hsic) + '\npps : ' + str(sine_pps) + ' , nmi : ' + str(sine_nmi))

plt.subplot(154)
plt.plot(para_data[:,0], para_data[:,1], 'bx')
plt.title('$\\rho$ : ' + str(para_pc) + ' , HSIC : ' + str(para_hsic) + '\npps : ' + str(para_pps) + ' , nmi : ' + str(para_nmi))

plt.subplot(155)
plt.plot(unif_data[:,0], unif_data[:,1], 'bx')
plt.title('$\\rho$ : ' + str(unif_pc) + ' , HSIC : ' + str(unif_hsic) + '\npps : ' + str(unif_pps) + ' , nmi : ' + str(unif_nmi))

plt.tight_layout()
plt.show()


