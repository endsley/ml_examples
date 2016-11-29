
import numpy as np
from sklearn.cluster import KMeans

def K_means(db):
	try:
		db['allocation'] = KMeans(n_clusters= db['C_num']).fit_predict(db['normalized_U_matrix'])
		db['allocation'] += 1		# starts from 1 instead of 0
	
		db['binary_allocation'] = np.zeros( ( db['normalized_U_matrix'].shape[0], db['C_num'] ) )
	
		#	Convert from allocation to binary_allocation
		for m in range(db['allocation'].shape[0]):
			db['binary_allocation'][m, db['allocation'][m] - 1 ] = 1
	
		if db['Y_matrix'].shape[0] == 0:
			db['Y_matrix'] = db['binary_allocation']
		else:
			db['Y_matrix'] = np.append( db['Y_matrix'] , db['binary_allocation'], axis=1)
	except:
		import pdb;
		pdb.set_trace()

