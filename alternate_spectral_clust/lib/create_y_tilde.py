
import numpy as np

def create_y_tilde(db):
	Kernel_y = np.dot(db['Y_matrix'], np.transpose(db['Y_matrix']))
	inner_p = np.dot(np.dot(db['H_matrix'], Kernel_y), db['H_matrix'])
	return db['D_matrix'].dot(inner_p).dot(db['D_matrix'])

