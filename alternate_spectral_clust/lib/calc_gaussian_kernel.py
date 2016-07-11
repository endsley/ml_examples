

import numpy as np

def calc_gaussian_kernel(db):
	big_W = np.dot(db['W_matrix'], np.transpose(db['W_matrix']))

	for i in range(db['N']):
		for j in range(db['N']):
			x_dif = db['data'][i] - db['data'][j]
			x_dif = x_dif[np.newaxis]

			db['Kernel_matrix'][i,j] = np.dot(np.dot( x_dif, big_W), np.transpose(x_dif))

	db['Kernel_matrix'] = np.exp(-db['Kernel_matrix']/(2*np.power(db['sigma'],2)))
	db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

#	db['Kernel_matrix'] = np.round(db['Kernel_matrix'],20)
