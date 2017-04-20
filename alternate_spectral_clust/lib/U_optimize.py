
import numpy as np
import large_eig

def U_optimize(db) :

	L = np.dot(np.dot(db['D_matrix'], db['Kernel_matrix']), db['D_matrix'])
	L = db['H_matrix'].dot(L).dot(db['H_matrix'])

	if L.shape[0]*L.shape[1] > 3000*3000:
		eigenVectors, eigenValues = large_eig.nystrom(L, db['C_num'], 0.2)
	else :
		eigenValues,eigenVectors = np.linalg.eigh(L)

	idx = eigenValues.argsort()
	idx = idx[::-1]
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]

	db['U_matrix'] = eigenVectors[:,:db['C_num']]


	if db['W_matrix'].size == db['d']*db['d']: # W should always be dxq except during initialization
		W = np.zeros((db['d'], db['q']) )

	#print 'UUUUUU : ' , db['U_matrix']
