#!/usr/bin/python

import numpy as np
from numpy import genfromtxt

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)

data = genfromtxt('load.csv', delimiter=',')


means = np.round(np.mean(data, axis=0),3)
SD = np.round(np.std(data, axis=0), 3)
maxes = np.round(np.max(data, axis=0),3)
mines = np.round(np.min(data, axis=0),3)
sumes = np.round(np.sum(data, axis=0),3)

nmi_quality_mean = means[0]
nmi_quality_best = maxes[0]
nmi_quality_var = SD[0]

nmi_alternative_mean = means[1]
nmi_alternative_best = mines[1]
nmi_alternative_var = SD[1]

cluster_quality_mean = means[2]
cluster_quality_SD = SD[2]

cost_mean = means[3]
cost_SD = SD[3]

cost_best = mines[3]
time_mean = means[4]
time_std = SD[4]
total_time = sumes[4]

print 'NMI : ', nmi_quality_mean , '+' , nmi_quality_var
print 'NMI max : ', nmi_quality_best
print 'Alternative NMI : ', nmi_alternative_mean , '+' , nmi_alternative_var
print 'Cluster Q : ', cluster_quality_mean , '+' , cluster_quality_SD
print 'Cost : ', cost_mean , '+' , cost_SD
print 'Time : ', time_mean , '+' , time_std

#print '\n'
#print nmi_quality_mean, '\t' , nmi_quality_best, '\t' , nmi_quality_var, '\t',
#print nmi_alternative_mean, '\t', nmi_alternative_best, '\t' , nmi_alternative_var , '\t',
#print cluster_quality_mean, '\t', cost_mean , '\t', total_time , '\n\n'


#import pdb; pdb.set_trace()
