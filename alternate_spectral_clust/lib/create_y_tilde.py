
import numpy as np

#	In this y_tilde, instead of using tr(H K_y H K), I decided to also normalize the 2nd K matrix
#	instead, I have tr(H K_y H D^-1/2 K D^-1/2) = tr(D^-1/2 H K_y H D^-1/2 K) = tr(y_tilde K)
#	I have redefined y_tilde = D^-1/2 H K_y H D^-1/2
#	During the calculation of the gradient descend, the kernel is NOT normalized
#
#	The reason why I put it here is b/c normalized kernel is more numerically stable

def create_y_tilde(db):
	Kernel_y = np.dot(db['Y_matrix'], np.transpose(db['Y_matrix']))
	inner_p = np.dot(np.dot(db['H_matrix'], Kernel_y), db['H_matrix'])
	return db['D_matrix'].dot(inner_p).dot(db['D_matrix'])

