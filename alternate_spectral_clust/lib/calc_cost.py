from create_gamma_ij import *
		

def calc_cost_function(self, db, W):
	#	Calculate dL/dw gradient
	Z = db['Z_matrix']
	L1 = db['L1']
	L2 = db['L2']
	iv_all = np.array(range(db['N']))
	jv_all = iv_all

	Z_shape = Z.shape
	I = np.eye(Z_shape[1])

	#	Calc Base
	cost = 0
	for i in iv_all:
		for j in jv_all:
			
			x_dif = db['data'][i] - db['data'][j]
			x_dif = x_dif[np.newaxis]

			gamma_ij = create_gamma_ij(db, db['y_tilde'], i, j)
			cost = cost -  gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))


	term1 = W.T.dot(Z) - I
	term2 = W - Z

	Lagrange = np.trace(L1.dot(term1)) + np.trace(L2.dot(term2))
	Aug_lag = np.sum(term1*term1) + np.sum(term2*term2)

	Lagrange_cost = cost + Lagrange + Aug_lag
	return [Lagrange_cost, cost]



