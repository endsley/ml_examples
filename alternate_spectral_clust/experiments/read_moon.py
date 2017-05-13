#!/usr/bin/python

import numpy as np

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)


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

for sampN, eachDic in All_dic.items():
	eachDic['ISM'] = np.array(eachDic['ISM']).astype(np.float)
	eachDic['SM'] = np.array(eachDic['SM']).astype(np.float)
	eachDic['DG'] = np.array(eachDic['DG']).astype(np.float)

	#j
	#All_dic
	#print i




#for i,j in All_dic[200].items():
#	print i
#	for m in j:
#		print '\t', m

import pdb; pdb.set_trace()
