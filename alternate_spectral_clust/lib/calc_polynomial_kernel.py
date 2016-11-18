

import numpy as np

def calc_polynomial_kernel(db):
	big_W = np.dot(db['W_matrix'], np.transpose(db['W_matrix']))

	for i in range(db['N']):
		for j in range(db['N']):
			x_i = db['data'][i] 
			x_j = db['data'][j]
			x_j = np.transpose(x_j[np.newaxis])

			db['Kernel_matrix'][i,j] = np.power(np.dot(np.dot( x_i, big_W), x_j) + db['polynomial_constant'] , db['poly_order'])


	db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

