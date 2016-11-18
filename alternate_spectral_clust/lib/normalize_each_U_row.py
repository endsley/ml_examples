

import numpy as np

def normalize_each_U_row(db):
	normV = np.linalg.norm(db['U_matrix'], ord=2, axis=1, keepdims=True)
	divide_mat = np.matlib.repmat(normV, 1, db['U_matrix'].shape[1])
	db['normalized_U_matrix'] = db['U_matrix'] / divide_mat


