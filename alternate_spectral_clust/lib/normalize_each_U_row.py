
import pdb
import numpy as np
from sklearn.preprocessing import normalize

def normalize_each_U_row(db):
	db['normalized_U_matrix'] = normalize(db['U_matrix'], norm='l2', axis=1)


	#	This is the older version which i wrote
	#normV = np.linalg.norm(db['U_matrix'], ord=2, axis=1, keepdims=True)
	#divide_mat = np.matlib.repmat(normV, 1, db['U_matrix'].shape[1])
	#db['normalized_U_matrix'] = db['U_matrix'] / divide_mat
	#pdb.set_trace()


