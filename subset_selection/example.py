#!/usr/bin/env python

from subset_select import *


SS = subset_select('../dataset/breast-cancer.csv')
[new_X, best_test_sample_id] = SS.get_subset()

#breast-cancer-labels.csv
