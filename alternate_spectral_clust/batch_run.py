#!/usr/bin/python

import os
import sys
sys.path.append('./lib')
from file_writing import *



found_good_solution = False
run_id = 0
while not found_good_solution:
	#txt = 'Run ' + str(run_id) + '\n'
	#append_txt('./output.txt', txt)
	os.system("./main.py") 
	#run_id += 1
