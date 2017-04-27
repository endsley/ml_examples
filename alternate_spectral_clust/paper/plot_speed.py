#!/usr/bin/python

import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 

x = [1, 2, 3, 4, 5, 6]
run_times = 1
KDAC = np.log(run_times*np.array([0.71, 295, 516, 909.6, 894, 29088]))
ISM = np.log(np.array([0.01, 1.7, 3.6, 0.35, 0.78, 18.56]))
GPU = np.log(run_times*np.array([0.023, 12.67, 6.49, 3.8, 134, 600]))
Orthogonal = np.log(run_times*np.array([0.07, 2.34, 3.4, 0.3, 0.68, 16.73]))
labels = ['4 Small\nGaussians', 'Escher\nFlower', '4 Large\nGaussians', 'Moon no\nnoise', 'Moon with\nnoise', 'Face image\ndata']

plt.subplot(121)
plt.plot(x, KDAC, 'r-'); plt.plot(x, KDAC, 'ro');
plt.plot(x, ISM, 'b-.'); plt.plot(x, ISM, 'bo');
plt.plot(x, GPU, 'y--'); plt.plot(x, GPU, 'yo');
plt.plot(x, Orthogonal, 'g:'); plt.plot(x, Orthogonal, 'go');

plt.text(6, KDAC[5], 'KDAC')
plt.text(6, GPU[5], 'KDAC GPU')
plt.text(6, ISM[5]-1, 'ISM')
plt.text(6, Orthogonal[5], 'Orthogonal')

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.ylabel('ln(Time (s))')
plt.title('Experiment Vs Time of single execusion (s)')


# --------------------------------

run_times = 30
KDAC = np.log(run_times*np.array([0.71, 295, 516, 909.6, 894, 29088]))
ISM = np.log(np.array([0.01, 1.7, 3.6, 0.35, 0.78, 18.56]))
GPU = np.log(run_times*np.array([0.023, 12.67, 6.49, 3.8, 134, 600]))
Orthogonal = np.log(run_times*np.array([0.07, 2.34, 3.4, 0.3, 0.68, 16.73]))
labels = ['4 Small\nGaussians', 'Escher\nFlower', '4 Large\nGaussians', 'Moon no\nnoise', 'Moon with\nnoise', 'Face image\ndata']

plt.subplot(122)
plt.plot(x, KDAC, 'r-'); plt.plot(x, KDAC, 'ro');
plt.plot(x, ISM, 'b-.'); plt.plot(x, ISM, 'bo');
plt.plot(x, GPU, 'y--'); plt.plot(x, GPU, 'yo');
plt.plot(x, Orthogonal, 'g:'); plt.plot(x, Orthogonal, 'go');

plt.text(6, KDAC[5], 'KDAC')
plt.text(6, GPU[5], 'KDAC GPU')
plt.text(6, ISM[5]-1, 'ISM')
plt.text(6, Orthogonal[5], 'Orthogonal')

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.ylabel('ln(Time (s))')
plt.title('Experiment Vs Time of total execusion (s)')


plt.show()
