
import numpy as np
from create_gamma_ij import *
from create_y_tilde import *
from objective_magnitude import *
from exponential_solver import *



def W_optimize_Gaussian(db):
	y_tilde = create_y_tilde(db)

	db['Z_matrix'] = db['W_matrix']
	db['L1'] = np.eye(db['q'])
	db['L2'] = np.eye(db['q'], db['d'])

	db['L'] = np.append(db['L1'], db['L2'], axis=1)	
	
	eSolver = exponential_solver(db, y_tilde)
	
	use_all_data = False
	if use_all_data :
		iv = np.array(range(db['N']))
		jv = iv
	else:
		i_values = np.random.permutation( np.array(range(db['N'])) )
		iv = i_values[0:db['SGD_size']]
		j_values = np.random.permutation( np.array(range(db['N'])) )
		jv = j_values[0:db['SGD_size']]

	db['W_matrix'] = eSolver.run(iv,jv)

	print db['W_matrix']
	import pdb; pdb.set_trace();

