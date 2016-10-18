from create_gamma_ij import *
		

def calc_cost_function(db, W):
	#	Calculate dL/dw gradient
	iv_all = np.array(range(db['N']))
	jv_all = iv_all


	#	Calc Base
	cost = 0
	for i in iv_all:
		for j in jv_all:
			
			x_dif = db['data'][i] - db['data'][j]
			x_dif = x_dif[np.newaxis]

			gamma_ij = create_gamma_ij(db, db['y_tilde'], i, j)
			cost = cost -  gamma_ij*np.exp(-x_dif.dot(W).dot(W.T).dot(x_dif.T))

	return cost

