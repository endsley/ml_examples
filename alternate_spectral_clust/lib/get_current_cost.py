
from create_gamma_ij import *

def get_current_cost(db):
	W = db['W_matrix']
	iv = np.array(range(db['N']))
	jv = iv


	I = np.eye(W.shape[1])

	#	Setting up the cost function
	cost_foo = 0
	for i in iv:
		for j in jv:
			
			x_dif = db['data'][i] - db['data'][j]
			x_dif = x_dif[np.newaxis]
		
			gamma_ij = create_gamma_ij(db, db['y_tilde'], i, j)
			cost_foo = cost_foo - gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

	return cost_foo

