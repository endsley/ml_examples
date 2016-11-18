
import numpy as np
from U_optimize import *

def optimize_linear_kernel(db):

	if db['prev_clust'] == 0: db['W_matrix'] = np.identity(db['d'])
	loop_count = 0

	WU_converge = False
	while WU_converge == False: 	
		feature_space = db['data'].dot(db['W_matrix'])
		db['Kernel_matrix'] = feature_space.dot(feature_space.transpose())
		db['D_matrix'] = 1
	
		U_optimize(db)

		if db['prev_clust'] == 0:
			return

		# A = X'DH(UU'-YY')HDX 		H and D are symmetric
		right_side = db['H_matrix'].dot(db['D_matrix']).dot(db['data'])
		middle = db['U_matrix'].dot(db['U_matrix'].transpose()) - db['Y_matrix'].dot(db['Y_matrix'].transpose()) 
		L = (right_side.transpose()).dot(middle).dot(right_side)
	
		#	Get EigenVectors and Values
		eigenValues,eigenVectors = np.linalg.eigh(L)
		idx = eigenValues.argsort()
		idx = idx[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]	
		db['W_matrix'] = eigenVectors[:,:db['q']]
	
		#db['U_matrix'] = np.dot(db['data'], db['W_matrix'])


		if not db.has_key('previous_U_matrix'): 
			db['previous_U_matrix'] = db['U_matrix']
			db['previous_W_matrix'] = db['W_matrix']
		else:
			matrix_mag = np.linalg.norm(db['U_matrix'])
			U_change = np.linalg.norm(db['previous_U_matrix'] - db['U_matrix'])
			W_change = np.linalg.norm(db['previous_W_matrix'] - db['W_matrix'])

			if (U_change + W_change)/matrix_mag < 0.001: WU_converge = True


		db['previous_U_matrix'] = db['U_matrix']
		db['previous_W_matrix'] = db['W_matrix']
		loop_count += 1

		if loop_count > 50:
			WU_converge = True


