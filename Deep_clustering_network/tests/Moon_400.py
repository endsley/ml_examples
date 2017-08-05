#!/usr/bin/python

from DCN import *
import numpy as np
from numpy import genfromtxt



#	load data
data = genfromtxt('datasets/moon_400_2.csv', delimiter=',')

dcn = DCN(data,2, 'moon_400_2')
dcn.run()
