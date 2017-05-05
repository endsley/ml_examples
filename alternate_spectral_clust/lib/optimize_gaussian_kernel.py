
import numpy as np
from calc_gaussian_kernel import *
from U_optimize import *
from objective_magnitude import *
from cost_function import *
import time 
from orthogonal_optimization import *
import pickle
import os

#	You must comment out one of the method and keep the other
#from W_optimize_Gaussian import *
#from W_optimize_Gaussian_ADMM import *
from W_optimize_Gaussian_stochastic import *
from SDG import *
#from direct_GD import *


def Update_latest_UW(db):
	if np.linalg.norm(db['W_matrix']) == 0: return

	new_cost = db['cf'].calc_cost_function(db['W_matrix'])

	#import pdb; pdb.set_trace()
	if(new_cost < db['lowest_cost']):
		db['lowest_U'] = db['U_matrix']
		db['lowest_cost'] = new_cost

def exit_condition(db, loop_count):	
	if not db.has_key('previous_U_matrix'): 
		db['previous_U_matrix'] = db['U_matrix']
		db['previous_W_matrix'] = db['W_matrix']
	else:
		matrix_mag = np.linalg.norm(db['U_matrix'])
		U_change = np.linalg.norm(db['previous_U_matrix'] - db['U_matrix'])/matrix_mag
		W_change = np.linalg.norm(db['previous_W_matrix'] - db['W_matrix'])/np.linalg.norm(db['W_matrix'])

		#print '\t\tU change ratio : ' , U_change/matrix_mag
		#print '\t\tW change ratio : ' , W_change/np.linalg.norm(db['W_matrix'])

		if U_change < 0.001 and W_change < 0.001: return True
			
	db['previous_U_matrix'] = db['U_matrix']
	db['previous_W_matrix'] = db['W_matrix']
	
	if loop_count > db['maximum_U_update_count']: return True

	return False

def report_current_status(db):
	minutes = (time.time() - db['start_time'])/60.0
	print '\n\nAt minute : ', minutes , '\n\n'


def properly_initialize_U(db):
	db['W_matrix'] = np.identity(db['d'])
	if db['data_type'] == 'Feature Matrix': 
		db['Kernel_matrix'] = db['cf'].create_Kernel(db['W_matrix'])
	elif db['data_type'] == 'Graph matrix': 
		db['Kernel_matrix'] = db['data']
		db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)

	U_optimize(db)

def save_initial_W(W_matrix):
	if os.path.exists("./init_W.pk"):
		init_W = pickle.load( open( "init_W.pk", "rb" ) )
	else:
		init_W = []

	init_W.append(W_matrix)
	pickle.dump( init_W, open( "init_W.pk", "wb" ) )



def Orthogonal_implementation(db):
	cf = cost_function(db)
	db['cf'] = cf
	OO = orthogonal_optimization(cf.calc_cost_function, cf.calc_gradient_function)

	if db['prev_clust'] == 0: 
		properly_initialize_U(db)
	else:
		if db['U_matrix'].size == 0: properly_initialize_U(db)

		WU_converge = False
		loop_count = 0
		db['lowest_cost'] = float("inf")
		db['lowest_gradient'] = float("inf")

		#	Generate W
		if False:												#Use identity
			db['W_matrix'] = np.eye(db['d'], db['q']) 		
		else:
			W_temp = np.random.randn(db['d'], db['q']) 			# randomize initialization
			[Q,R] = np.linalg.qr(W_temp)
			db['W_matrix'] = Q
			save_initial_W(db['W_matrix'])						# each random initialization, it is stored here
			print Q[0:2,:]

		db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
	
		while WU_converge == False: 	
			cf.calc_psi()	# need a better way to initializating
			Update_latest_UW(db)

			db['W_matrix'] = OO.run(db['W_matrix'], db['maximum_W_update_count'])		# Use a lower max_rep if it doesn't finish running	
			U_optimize(db)
	
			WU_converge = exit_condition(db, loop_count)
			loop_count += 1

			report_current_status(db)
		cf.calc_psi()	# this make sure that cost function are accurate

def ISM_implementation(db):
	cf = cost_function(db)
	db['cf'] = cf
	#OO = orthogonal_optimization(cf.calc_cost_function, cf.calc_gradient_function)

	if db['prev_clust'] == 0: 
		properly_initialize_U(db)
	else:
		if db['U_matrix'].size == 0: properly_initialize_U(db)

		WU_converge = False
		loop_count = 0
		db['lowest_cost'] = float("inf")
		db['lowest_gradient'] = float("inf")

		db['W_matrix'] = np.zeros((db['d'], db['q']) )

		if db['data_type'] == 'Feature Matrix': 
			db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
		elif db['data_type'] == 'Graph matrix': 
			db['Kernel_matrix'] = db['data']
			db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)
	
		Toggle = True
		while WU_converge == False: 	
			cf.calc_psi()	# need a better way to initializating
			Update_latest_UW(db)

			W_optimize_Gaussian_SDG(db)
			U_optimize(db)
	
			WU_converge = exit_condition(db, loop_count)
			loop_count += 1
			report_current_status(db)

		cf.calc_psi()	# this make sure that cost function are accurate

def DongLing_implementation(db):
	cf = cost_function(db)
	db['cf'] = cf

	if db['prev_clust'] == 0: 
		properly_initialize_U(db)
	else:
		if db['U_matrix'].size == 0: properly_initialize_U(db)

		WU_converge = False
		loop_count = 0
		db['lowest_cost'] = float("inf")
		db['lowest_gradient'] = float("inf")

		if False: # running both
			if db['Y_matrix'].size > 0:
				db['Y_matrix'] = db['Y_matrix'][:,0:db['C_num']]
		elif True:	# If we initialize W from some pickle file
			if os.path.exists("./init_W.pk"):
				init_W = pickle.load( open( "init_W.pk", "rb" ) )
				db['W_matrix'] = init_W[9]
				print init_W[9]
			else:
				db['W_matrix'] = np.eye(db['d'], db['q']) 
		else:
			db['W_matrix'] = np.eye(db['d'], db['q']) #This must be commented out if running together with FKDAC

		if db['data_type'] == 'Feature Matrix': 
			db['Kernel_matrix'] = cf.create_Kernel(db['W_matrix'])
		elif db['data_type'] == 'Graph matrix': 
			db['Kernel_matrix'] = db['data']
			db['D_matrix'] = np.diag(1/np.sqrt(np.sum(db['Kernel_matrix'],axis=1))) # 1/sqrt(D)
	
		while WU_converge == False: 	
			W_optimize_Gaussian(db)
			U_optimize(db)
	
			WU_converge = exit_condition(db, loop_count)
			loop_count += 1
			report_current_status(db)


def optimize_gaussian_kernel(db):
	db['start_time'] = time.time() 

	#ISM_implementation(db)
	DongLing_implementation(db)
	#Orthogonal_implementation(db)

