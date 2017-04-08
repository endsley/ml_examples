#!/usr/bin/python

import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
#	WebKB results (Hours)
ISM = np.array([[0,0],[15.4, 0.362]])
DG = np.array([[0,0],[80, 0.01]])
Orthogonal = np.array([[0,0],[31971,0.096],[28685,0.098],[43246,0.087],[39912,0.081],[28506,0.11],[23878,0.078],[32350, 0.082],[27134,0.11],[41599,0.078]])
Orthogonal[:,0] = np.cumsum(Orthogonal[:,0])/(60.0*60.0)	# change time to hours

plt.subplot(122)
plt.plot(ISM[:,0], ISM[:,1], 'b-'); plt.plot(ISM[:,0], ISM[:,1], 'bo');
plt.plot(Orthogonal[:,0], Orthogonal[:,1], 'r-'); plt.plot(Orthogonal[:,0], Orthogonal[:,1], 'ro');
plt.plot(DG[:,0], DG[:,1], 'g-'); plt.plot(DG[:,0], DG[:,1], 'go');

plt.text(ISM[1,0], ISM[1,1], 'ISM')
plt.text(Orthogonal[Orthogonal.shape[0]-1,0]-10, Orthogonal[Orthogonal.shape[0]-1,1], 'Orthogonal\nMethod')
plt.text(DG[DG.shape[0]-1,0]-13, DG[DG.shape[0]-1,1]+0.01, 'DG/DG GPU not\nyet finished')

plt.xlabel('Time (Hours)')
plt.ylabel('NMI result against ground truth')
plt.title('WebKB data Time(H) vs NMI quality\ndata is 1041x140')

# --------------------------------------------

#	Faces results (Min)
plt.subplot(121)
ISM = np.array([[0,0],[18.37, 0.44]])
DG = np.array([[0,0],[3*60, 0.005]])
Orthogonal = np.array([[0,0],[900,0.19],[933.93,0.36],[977,0.41],[995,0.33],[988,0.34],[823, 0.45],[833,0.31],[916,0.24],[916,0.47],[961,0.29]])
Orthogonal[:,0] = np.cumsum(Orthogonal[:,0])/(60.0)	# change time to minutes

plt.plot(ISM[:,0], ISM[:,1], 'b-'); plt.plot(ISM[:,0], ISM[:,1], 'bo');
plt.plot(Orthogonal[:,0], Orthogonal[:,1], 'r-'); plt.plot(Orthogonal[:,0], Orthogonal[:,1], 'ro');
plt.plot(DG[:,0], DG[:,1], 'g-'); plt.plot(DG[:,0], DG[:,1], 'go');

plt.text(ISM[1,0]-0.05, ISM[1,1], 'ISM')
plt.text(Orthogonal[Orthogonal.shape[0]-1,0]-20, Orthogonal[Orthogonal.shape[0]-1,1]-0.05, 'Orthogonal\nMethod')
plt.text(DG[DG.shape[0]-1,0]-23, DG[DG.shape[0]-1,1]+0.01, 'DG/DG GPU not\nyet finished')

plt.xlabel('Time (Minutes)')
plt.ylabel('NMI result against ground truth')
plt.title('Face data Time(M) vs NMI quality\ndata is 624x27')

plt.show()
