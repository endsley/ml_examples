#!/usr/bin/python


"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

run_times = 30
KDAC = np.log(run_times*np.array([0.71, 295, 516, 909.6, 894, 29088]))
ISM = np.log(np.array([0.01, 1.7, 3.6, 0.35, 0.78, 18.56]))
GPU = np.log(run_times*np.array([0.023, 12.67, 6.49, 3.8, 134, 600]))
Orthogonal = np.log(run_times*np.array([0.07, 2.34, 3.4, 0.3, 0.68, 16.73]))

#minMag = np.min(np.hstack((KDAC, ISM))) - 1
#maxMag = np.max(np.hstack((KDAC, ISM))) + 1
##print 'Min : ' , np.min(np.hstack((KDAC, ISM))) - 1
#
#
#KDAC = KDAC + np.abs(minMag)
#ISM = ISM + np.abs(minMag)




N = len(KDAC)
#men_std = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars

plt.subplot(122)
#plt.bar(ind, KDAC, width, color='w', hatch="/" )
rects1 = plt.bar(ind, KDAC, width, color='w', hatch="/" )
rects2 = plt.bar(ind + width, GPU, width, color='w', hatch='*')
rects3 = plt.bar(ind + 2*width, Orthogonal, width, color='w', hatch='x')
rects4 = plt.bar(ind + 3*width, ISM, width, color='w')

# add some text for labels, title and axes ticks
plt.ylabel('Log(Time) Seconds')
plt.title('30 Initialization Restarts Except ISM')
plt.xticks(ind + (width*4) / 2, ('S 4G', 'L 4G', 'Flower', 'Moon', 'Moon+noise', 'Face'))

#plt.ylim([minMag,maxMag])

plt.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('DG', 'GPU', 'OM', 'ISM'), loc = 'best')


plt.subplot(121)

KDAC = np.log(np.array([0.71, 295, 516, 909.6, 894, 29088]))
ISM = np.log(np.array([0.01, 1.7, 3.6, 0.35, 0.78, 18.56]))
GPU = np.log(np.array([0.023, 12.67, 6.49, 3.8, 134, 600]))
Orthogonal = np.log(np.array([0.07, 2.34, 3.4, 0.3, 0.68, 16.73]))

rects1 = plt.bar(ind, KDAC, width, color='w', hatch="/" )
rects2 = plt.bar(ind + width, GPU, width, color='w', hatch='*')
rects3 = plt.bar(ind + 2*width, Orthogonal, width, color='w', hatch='x')
rects4 = plt.bar(ind + 3*width, ISM, width, color='w')

# add some text for labels, title and axes ticks
plt.ylabel('Log(Time) Seconds')
plt.title('Single Run for All Techniques')
plt.xticks(ind + (width*4) / 2, ('S 4G', 'L 4G', 'Flower', 'Moon', 'Moon+noise', 'Face'))

#plt.ylim([minMag,maxMag])

plt.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('DG', 'GPU', 'OM', 'ISM'), loc = 'best')


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.80*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()
