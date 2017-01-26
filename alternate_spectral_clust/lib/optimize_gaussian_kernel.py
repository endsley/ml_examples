
import numpy as np
from calc_gaussian_kernel import *
from U_optimize import *
from objective_magnitude import *
from cost_function import *
import time 

#	You must comment out one of the method and keep the other
#from W_optimize_Gaussian import *
#from W_optimize_Gaussian_stochastic import *
#from W_optimize_Gaussian_ADMM import *
from SDG import *
#from direct_GD import *


def Update_latest_UW(db):
	new_cost = db['cf'].calc_cost_function(db['W_matrix'])

	#import pdb; pdb.set_trace()
	if(new_cost < db['lowest_cost']):
		db['lowest_U'] = db['U_matrix']
		db['lowest_cost'] = new_cost

def optimize_gaussian_kernel(db):
	WU_converge = False
	N = db['N']

	#if db['W_matrix'].shape[0] == 0:
	#	db['W_matrix'] = np.identity(db['d'])
	#else:
	#	db['W_matrix'] = db['W_matrix'][:,0:db['q']]
	W = np.random.normal(0,1, (db['d'], db['q']) )
	db['W_matrix'], r = np.linalg.qr(W)



	loop_count = 0
	db['lowest_cost'] = float("inf")
	db['lowest_gradient'] = float("inf")

	cf = cost_function(db)
	db['cf'] = cf

	while WU_converge == False: 	
		if db['data_type'] == 'Feature Matrix': 
			db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
		elif db['data_type'] == 'Graph matrix': 
			db['Kernel_matrix'] = db['data']
			db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

		U_optimize(db)
		if db['prev_clust'] == 0: return


		cf.calc_psi()	# need a better way to initializating
		Update_latest_UW(db)

		W_optimize_Gaussian_SDG(db)
		#W_optimize_Gaussian(db)
		

	
		if not db.has_key('previous_U_matrix'): 
			db['previous_U_matrix'] = db['U_matrix']
			db['previous_W_matrix'] = db['W_matrix']
		else:
			matrix_mag = np.linalg.norm(db['U_matrix'])
			U_change = np.linalg.norm(db['previous_U_matrix'] - db['U_matrix'])/matrix_mag
			W_change = np.linalg.norm(db['previous_W_matrix'] - db['W_matrix'])/np.linalg.norm(db['W_matrix'])

			#print '\t\tU change ratio : ' , U_change/matrix_mag
			#print '\t\tW change ratio : ' , W_change/np.linalg.norm(db['W_matrix'])

			if U_change < 0.001 and W_change < 0.001: 
				#db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
				#db['D_matrix'] = cf.create_D_matrix(db['Kernel_matrix'])
				#U_optimize(db)
				WU_converge = True


		db['previous_U_matrix'] = db['U_matrix']
		db['previous_W_matrix'] = db['W_matrix']
		loop_count += 1
		
		#print db['updated_magnitude']
		#print 'Loop count = ' , loop_count
		if loop_count > 10:
			#db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
			#db['D_matrix'] = cf.create_D_matrix(db['Kernel_matrix'])
			#U_optimize(db)
			WU_converge = True

		#print '\nAfter U cost : ' , cf.calc_cost_function(db['W_matrix'])



	#db['U_matrix'] = db['lowest_U']
	#print 'Major cost : ' , cf.calc_cost_function(db['W_matrix'])
