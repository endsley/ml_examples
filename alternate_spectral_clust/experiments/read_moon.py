#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


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


plotDic = {}
plotDic['ISM'] = np.empty((0, 5))
plotDic['SM']  = np.empty((0, 5))
plotDic['DG']  = np.empty((0, 5))

#V = np.empty((3, 0))
#V = np.hstack((V, f))

for sampN, eachDic in All_dic.items():
	eachDic['ISM'] = np.array(eachDic['ISM']).astype(np.float)[0]
	eachDic['SM'] = np.array(eachDic['SM']).astype(np.float)
	eachDic['DG'] = np.array(eachDic['DG']).astype(np.float)

	ISM_sample = eachDic['ISM'][0]
	ISM_time = eachDic['ISM'][1]
	ISM_quality = eachDic['ISM'][4]
	ISM_row = np.array([ISM_sample, np.log(ISM_time), 0, ISM_quality, 0])
	plotDic['ISM'] = np.vstack((plotDic['ISM'], ISM_row))


	SM_mean = np.mean(eachDic['SM'], axis=0)
	SM_std = np.std(eachDic['SM'], axis=0)


	SM_sample = SM_mean[0]
	SM_time = SM_mean[1]
	SM_time_std = SM_std[1]
	SM_quality = SM_mean[4]
	SM_quality_std = SM_std[4]
	SM_row = np.array([SM_sample, np.log(SM_time), np.log(SM_time_std), SM_quality, SM_quality_std])
	plotDic['SM'] = np.vstack((plotDic['SM'], SM_row))


	DG_mean = np.mean(eachDic['DG'], axis=0)
	DG_std = np.std(eachDic['DG'], axis=0)

	DG_sample = DG_mean[0]
	DG_time = DG_mean[1]
	DG_time_std = DG_std[1]
	DG_quality = DG_mean[4]
	DG_quality_std = DG_std[4]
	DG_row = np.array([DG_sample, np.log(DG_time), np.log(DG_time_std), DG_quality, DG_quality_std])
	plotDic['DG'] = np.vstack((plotDic['DG'], DG_row))


plotDic['DG'] = plotDic['DG'][plotDic['DG'][:,0].argsort()]
plotDic['SM'] = plotDic['SM'][plotDic['SM'][:,0].argsort()]
plotDic['ISM'] = plotDic['ISM'][plotDic['ISM'][:,0].argsort()]

plt.subplot(121)
plt.errorbar(plotDic['SM'][:,0]+20, plotDic['SM'][:,1], yerr=plotDic['SM'][:,2], fmt='-o')
plt.errorbar(plotDic['ISM'][:,0], plotDic['ISM'][:,1], yerr=plotDic['ISM'][:,2], fmt='-o')
plt.errorbar(plotDic['DG'][:,0]-20, plotDic['DG'][:,1], yerr=plotDic['DG'][:,2], fmt='-o')

plt.text(plotDic['ISM'][-1,:][0]+30, plotDic['ISM'][-1,:][1], 'ISM')
plt.text(plotDic['SM'][-1,:][0]+30, plotDic['SM'][-1,:][1], 'SM')
plt.text(plotDic['DG'][-1,:][0]+30, plotDic['DG'][-1,:][1], 'DG')

plt.xlabel('Number of sample')
plt.ylabel('Single Run execution time log(seconds)')
plt.title('Moon Experiment, Sample Num vs log(Time)')
plt.axis((0,2200,-3,24))
#--------------------------------------
plt.subplot(122)
#plt.errorbar(plotDic['SM'][:,0]+20, plotDic['SM'][:,3], yerr=np.log10(plotDic['SM'][:,4]), fmt='-o')
#plt.errorbar(plotDic['ISM'][:,0], plotDic['ISM'][:,3], 	yerr=np.log10(plotDic['ISM'][:,4]), fmt='-o')
#plt.errorbar(plotDic['DG'][:,0]-20, plotDic['DG'][:,3], yerr=np.log10(plotDic['DG'][:,4]), fmt='-o')

plt.errorbar(plotDic['SM'][:,0]+20, plotDic['SM'][:,3], yerr=plotDic['SM'][:,4], fmt='-o')
plt.errorbar(plotDic['ISM'][:,0], plotDic['ISM'][:,3], 	yerr=plotDic['ISM'][:,4], fmt='-o')
plt.errorbar(plotDic['DG'][:,0]-20, plotDic['DG'][:,3], yerr=plotDic['DG'][:,4], fmt='-o')

#plt.errorbar(plotDic['SM'][:,0]+20, plotDic['SM'][:,3], fmt='-o')
#plt.errorbar(plotDic['ISM'][:,0], plotDic['ISM'][:,3], fmt='-o')
#plt.errorbar(plotDic['DG'][:,0]-20, plotDic['DG'][:,3], fmt='-o')


plt.text(plotDic['ISM'][-1,:][0]+30, plotDic['ISM'][-1,:][3], 'ISM')
plt.text(plotDic['SM'][-1,:][0]+30, plotDic['SM'][-1,:][3], 'SM')
plt.text(plotDic['DG'][-1,:][0]+30, plotDic['DG'][-1,:][3], 'DG')

plt.xlabel('Number of sample')
plt.ylabel('Average NMI Quality')
plt.title('Moon Experiment, Sample Num vs NMI Quality')
plt.axis((0,2200,-0.5,1.4))





plt.show()
import pdb; pdb.set_trace()





