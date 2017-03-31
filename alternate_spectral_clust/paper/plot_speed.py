#!/usr/bin/python

import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('KDAC', 'FKDAC', 'Orthogonal')
y_pos = np.arange(len(objects))
performance = [0.014, 0.44, 0.34]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
matplotlib.rc('xtick', labelsize=10) 
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('NMI')
plt.title('Alogrithm Vs NMI Clustering Quality against truth')
plt.show()

#
#x = [1, 2, 3, 4, 5]
#KDAC = np.array([1, 1, 1, 1, 0.48])
#FKDAC = np.array([1, 1, 1, 1, 0.44])
#Orthogonal = np.array([1, 1, 1, 1, 0.34])
#labels = ['4 Small\nGaussians', '4 Large\nGaussians', 'Moon no\nnoise', 'Moon with\nnoise', 'Face image\ndata']
#
#plt.plot(x, KDAC, 'r-')
#plt.plot(x, FKDAC, 'b-.')
#plt.plot(x, Orthogonal, 'g:')
#
#plt.text(5, KDAC[4], 'KDAC')
#plt.text(5, FKDAC[4], 'FKDAC')
#plt.text(5, Orthogonal[4], 'Orthogonal')
#

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.ylabel('ln(Time (s))')
plt.title('Experiment Vs Time of execusion (s)')
plt.show()



#-------------------------------------

#
#x = [1, 2, 3, 4, 5]
#KDAC = np.log(np.array([0.71, 516, 909.6, 894, 29088]))
#FKDAC = np.log(np.array([0.01, 3.6, 0.35, 0.78, 390]))
#GPU = np.log(np.array([0.023, 6.49, 3.8, 134, 600]))
#Orthogonal = np.log(np.array([0.07, 3.4, 0.3, 0.68, 17.3]))
#labels = ['4 Small\nGaussians', '4 Large\nGaussians', 'Moon no\nnoise', 'Moon with\nnoise', 'Face image\ndata']
#
#plt.plot(x, KDAC, 'r-')
#plt.plot(x, FKDAC, 'b-.')
#plt.plot(x, GPU, 'y--')
#plt.plot(x, Orthogonal, 'g:')
#
#plt.text(5, KDAC[4], 'KDAC')
#plt.text(5, GPU[4], 'KDAC GPU')
#plt.text(5, FKDAC[4], 'FKDAC')
#plt.text(5, Orthogonal[4], 'Orthogonal')
#
#
## You can specify a rotation for the tick labels in degrees or with keywords.
#plt.xticks(x, labels, rotation='vertical')
## Pad margins so that markers don't get clipped by the axes
#plt.margins(0.2)
## Tweak spacing to prevent clipping of tick-labels
#plt.subplots_adjust(bottom=0.15)
#plt.ylabel('ln(Time (s))')
#plt.title('Experiment Vs Time of execusion (s)')
#plt.show()
