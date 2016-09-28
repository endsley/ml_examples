
import numpy as np
from create_gamma_ij import *
from create_y_tilde import *

def objective_magnitude(db):

	part_1 = np.trace(db['U_matrix'].T.dot(db['D_matrix']).dot(db['Kernel_matrix']).dot(db['D_matrix']).dot(db['U_matrix']))
	part_2 = np.trace(db['H_matrix'].dot(db['Y_matrix']).dot(db['Y_matrix'].T).dot(db['H_matrix']).dot(db['Kernel_matrix']))

	import pdb; pdb.set_trace()

	print 'Part 1 : ' , part_1
	print 'Part 2 : ' , part_2
	print 'Lambda : ' , db['lambda']
	print 'Part 2 with Lambda: ' , part_2*db['lambda']

	return part_1 - part_2*db['lambda']

