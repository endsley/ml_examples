import numpy as np

def create_gamma_ij(db, y_tilde, i, j):
	degree_of_vertex = np.diag(db['D_matrix'])
	ith_row = db['U_matrix'][i,:]
	jth_row = db['U_matrix'][j,:]

	u_dot = np.dot(ith_row,jth_row)
	part_1 = u_dot*degree_of_vertex[i]*degree_of_vertex[j]

	gamma_ij = part_1 - db['lambda'] * y_tilde[i,j]
	return gamma_ij

