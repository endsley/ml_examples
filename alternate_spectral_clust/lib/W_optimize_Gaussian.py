
import numpy as np
from create_gamma_ij import *
from create_y_tilde import *
from objective_magnitude import *


def create_A_ij_matrix(db, i, j):
	x_dif = db['data'][i] - db['data'][j]
	x_dif = x_dif[np.newaxis]
	return np.dot(np.transpose(x_dif), x_dif)


def get_previous_wolfe_terms(db, w_l, y_tilde):
	if not db.has_key('previous_wolfe_direction'):
		[update_direction, updated_magnitude] = get_W_gradient(db, y_tilde, np.ones((db['N'],db['N'])), w_l)
		db['previous_wolfe_direction'] = update_direction
		db['previous_wolfe_magnitude'] = updated_magnitude

	return [db['previous_wolfe_direction'], db['previous_wolfe_magnitude']]

def get_alpha_passing_wolfe(db, y_tilde, w_l, new_direction):
	wolfe_passed = False
	alpha = 1
	a1 = 0.0001
	a2 = 0.9
	loop_exit = 80
	count = 0
	while not wolfe_passed:	
		new_W = np.sqrt(1-alpha*alpha)*w_l + alpha*new_direction
		[update_direction, updated_magnitude] = get_W_gradient(db, y_tilde, np.ones((db['N'],db['N'])), new_W)
		[previous_wolfe_direction, previous_wolfe_magnitude] = get_previous_wolfe_terms(db, w_l, y_tilde)

		#if updated_magnitude < previous_wolfe_magnitude:
		#	print 'updated_magnitude : ' , updated_magnitude
		#	print 'previous : ' , previous_wolfe_magnitude #+ a1*alpha*np.linalg.norm(previous_wolfe_direction)

		#um = updated_magnitude
		#pwm = previous_wolfe_magnitude
		#tt = a1*alpha*np.linalg.norm(previous_wolfe_direction)

		sufficient_increase_condition = updated_magnitude > previous_wolfe_magnitude + a1*alpha*np.dot(new_direction, new_direction)
		curvature_condition = np.dot(new_direction, update_direction) < a2*np.dot(new_direction, new_direction)


		if sufficient_increase_condition and curvature_condition : 
			db['previous_wolfe_direction'] = update_direction
			db['previous_wolfe_magnitude'] = updated_magnitude
			print 'passed'
			return alpha

		#if not sufficient_increase_condition: 
		#	print alpha, 'sufficient_increase_condition'
		#	print '\tupdated_magnitude : ' , updated_magnitude
		#	print '\tprevious : ' , previous_wolfe_magnitude #+ a1*alpha*np.linalg.norm(previous_wolfe_direction)
		#	#import pdb; pdb.set_trace()
		#if not curvature_condition : print alpha, 'curvature_condition'

		count = count + 1
		if count > loop_exit:
			wolfe_passed = True
			print 'forced'
			db['previous_wolfe_direction'] = update_direction
			db['previous_wolfe_magnitude'] = updated_magnitude
			alpha = 0
			#import pdb; pdb.set_trace()
			return alpha

		alpha = alpha*0.8


def get_W_gradient(db, y_tilde, previous_gw, w_l):
	update_direction = 0
	updated_magnitude = 0

	for i in range(db['N']):
		for j in range(db['N']):
			A_ij = create_A_ij_matrix(db,i,j)
			gamma_ij = create_gamma_ij(db, y_tilde, i, j)

			part_1 = np.true_divide(np.dot(A_ij, w_l), np.power(db['sigma'],2))
			part_2 = np.exp(np.true_divide(np.dot(w_l, part_1),-2))

			portion_magnitude = gamma_ij*previous_gw[i,j]*part_2;

			ij_addition = -portion_magnitude*part_1
			updated_magnitude += portion_magnitude
			update_direction += ij_addition
	
	#print 'updated_magnitud : ' , updated_magnitude
	return [update_direction, updated_magnitude]

def update_previous_gw(db, w, previous_gw):
	for i in range(db['N']):
		for j in range(db['N']):
			A_ij = create_A_ij_matrix(db,i,j)

			part_1 = np.true_divide(np.dot(A_ij, w), np.power(db['sigma'],2))
			previous_gw[i,j] = previous_gw[i,j]*np.exp(np.true_divide(np.dot(w, part_1),-2))

	return previous_gw

def get_orthogonal_vector(db, m, input_vector):	
	count_down = m

	#	perform Gram Schmit
	while count_down != 0:
		count_down = count_down - 1
		w_prev = db['W_matrix'][:,count_down]
	
		projected_direction = (np.dot(w_prev,input_vector)/np.dot(w_prev, w_prev))*w_prev
		input_vector = input_vector - projected_direction	
	
	input_vector = input_vector/np.linalg.norm(input_vector)
	return input_vector


def W_optimize_Gaussian(db):
	y_tilde = create_y_tilde(db)
	#print 'y tilde \n', y_tilde
	
	previous_gw = np.ones((db['N'],db['N']))
	w_converged = False
	last_W = None

	for m in range(db['q']):
		db['W_matrix'][:,m] = get_orthogonal_vector(db, m, db['W_matrix'][:,m])
		counter = 0
		new_alpha = 0.5

		while not w_converged:
			w_l = db['W_matrix'][:,m]
			[update_direction, db['updated_magnitude']] = get_W_gradient(db, y_tilde, previous_gw, w_l)
			update_direction = get_orthogonal_vector(db, m+1, update_direction) # m+1 is to also remove the current dimension


			#new_alpha = get_alpha_passing_wolfe(db, y_tilde, w_l, update_direction)

			new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
			#import pdb; pdb.set_trace()
			
			[tmp_dir, new_mag] = get_W_gradient(db, y_tilde, previous_gw, new_W)
			#print db['updated_magnitude'], new_mag, new_alpha
			while new_mag < db['updated_magnitude']:
				new_alpha = new_alpha * 0.8
				if new_alpha > 0.00001 :
					new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
					[tmp_dir, new_mag] = get_W_gradient(db, y_tilde, previous_gw, new_W)
					#import pdb; pdb.set_trace()
				else:
					new_alpha = 0
					break

			if type(last_W) != type(None):
				relative_magnitude = np.linalg.norm(new_W)
				distance_change_since_last_update = np.linalg.norm(last_W - new_W)

				if (distance_change_since_last_update/relative_magnitude) < 0.001 : 
					w_converged = True
					try:
						db.pop('previous_wolfe_direction')
						db.pop('previous_wolfe_magnitude')
					except: pass

			#print new_W
			last_W = new_W/np.linalg.norm(new_W)
			db['W_matrix'][:,m] = last_W
			counter += 1
			if counter > db['maximum_W_update_count']: 
				print '\nExit due to maximum update reached'
				w_converged = True
				try:
					db.pop('previous_wolfe_direction')
					db.pop('previous_wolfe_magnitude')
				except: pass

		w_converged = False
		previous_gw = update_previous_gw(db, last_W, previous_gw)
		#print previous_gw

	db['W_matrix'] = db['W_matrix'][:,0:db['q']]
	#print db['W_matrix']
