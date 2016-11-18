
import numpy as np
from calc_polynomial_kernel import *
from U_optimize import *
from W_optimize_polynomial import *

def optimize_polynomial_kernel(db):
	WU_converge = False
	N = db['N']

	if db['W_matrix'].shape[0] == 0:
		db['W_matrix'] = np.identity(db['d'])
	else:
		db['W_matrix'] = db['W_matrix'][:,0:db['q']]

	
	loop_count = 0

	while WU_converge == False: 	
		calc_polynomial_kernel(db)
		U_optimize(db)

		if db['prev_clust'] == 0:
			return

		W_optimize_polynomial(db)


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
		
		#print db['updated_magnitude']
		print 'Loop count = ' , loop_count
		if loop_count > 30:
			WU_converge = True


