#!/usr/bin/python

from numpy import genfromtxt

fin = open('moon_all.txt','r')
lns = fin.readlines()

def input_line(All_dic, sample_n, algorithm, line_split):
	if algorithm not in All_dic[sample_n]:
		All_dic[sample_n][algorithm] = []

	del line_split[1]
	All_dic[sample_n][algorithm].append(line_split)
	return All_dic

All_dic = {}
for m in lns:
	line_split = m.split(',')

	sample_n = int(line_split[0])
	algorithm = line_split[1]

	if sample_n in All_dic:
		All_dic = input_line(All_dic, sample_n, algorithm, line_split)
	else:
		All_dic[sample_n] = {}
		All_dic = input_line(All_dic, sample_n, algorithm, line_split)

for i,j in All_dic[200].items():
	print i
	for m in j:
		print '\t', m

import pdb; pdb.set_trace()
