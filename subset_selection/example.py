#!/usr/bin/env python

from subset_select import *
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#SS = subset_select('../dataset/breast-cancer.csv')
#[new_X, best_test_sample_id] = SS.get_subset()
#SS.save_subset_to_file('cancer', Y='../dataset/breast-cancer-labels.csv')



SS = subset_select('../dataset/spiral_arm_validation.csv')
[new_X, best_test_sample_id] = SS.get_subset()

plt.subplot(121)
plt.plot(SS.X[:,0], SS.X[:,1], 'x'); 
plt.subplot(122)
plt.plot(new_X[:,0], new_X[:,1], 'x'); 
plt.show()
