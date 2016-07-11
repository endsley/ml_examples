
import numpy as np

def U_optimize(db) :

	L = np.dot(np.dot(db['D_matrix'], db['Kernel_matrix']), db['D_matrix'])
	L = db['H_matrix'].dot(L).dot(db['H_matrix'])

	eigenValues,eigenVectors = np.linalg.eigh(L)
	idx = eigenValues.argsort()
	idx = idx[::-1]
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]

	db['U_matrix'] = eigenVectors[:,:db['C_num']]

	#print 'UUUUUU : ' , db['U_matrix']
