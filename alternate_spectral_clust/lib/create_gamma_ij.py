import numpy as np

def create_gamma_ij(db, y_tilde, i, j):
	degree_of_vertex = np.diag(db['D_matrix'])
	ith_row = db['U_matrix'][i,:]
	jth_row = db['U_matrix'][j,:]

	u_dot = np.dot(ith_row,jth_row)
	part_1 = u_dot*degree_of_vertex[i]*degree_of_vertex[j]

#	if i == 0 and j == 3:
#		#print 'U :' , db['U_matrix']
#		print 'ith : ' , ith_row
#		print 'jth : ' , jth_row
#		print 'part 1 , lambda, tilde : ' , part_1, db['lambda'], y_tilde[i,j]
#		print degree_of_vertex
#		print ':::' , u_dot, degree_of_vertex[i], degree_of_vertex[j]

	gamma_ij = part_1 - db['lambda'] * y_tilde[i,j]
	return gamma_ij

