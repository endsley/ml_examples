
import numpy as np
from create_gamma_ij import *
from create_y_tilde import *
from objective_magnitude import *



def create_A_ij_matrix(db, i, j):
	x_dif = db['data'][i] - db['data'][j]
	x_dif = x_dif[np.newaxis]
	return np.dot(np.transpose(x_dif), x_dif)



def Stochastic_W_gradient(db, y_tilde, previous_gw, w_l, i_values, j_values):
	update_direction = 0
	updated_magnitude = 0

	for i in i_values:
		for j in j_values:
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
	db['y_tilde'] = y_tilde
	
	previous_gw = np.ones((db['N'],db['N']))
	w_converged = False
	last_W = None

	for m in range(db['q']):
		db['W_matrix'][:,m] = get_orthogonal_vector(db, m, db['W_matrix'][:,m])
		counter = 0
		new_alpha = 1
		print 'Starting a new dimension'
		while not w_converged:
			w_l = db['W_matrix'][:,m]

			i_values = np.random.permutation( np.array(range(db['N'])) )
			i_values = i_values[0:db['N']] #SGD_size
			j_values = np.random.permutation( np.array(range(db['N'])) )
			j_values = j_values[0:db['N']]

			[update_direction, db['updated_magnitude']] = Stochastic_W_gradient(db, y_tilde, previous_gw, w_l, i_values, j_values)
			update_direction = get_orthogonal_vector(db, m+1, update_direction) # m+1 is to also remove the current dimension

			#new_alpha = 1	# this is with back trace
			new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
			#import pdb; pdb.set_trace()
			
			[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)

			print 'Previous mag : ', db['updated_magnitude'], 'New mag : ' , new_mag, new_alpha
			while new_mag < db['updated_magnitude']:
				new_alpha = new_alpha * 0.4
				#print 'Alpha : ', new_alpha
				if new_alpha > 0.00001 :
					new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
					[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)
					#import pdb; pdb.set_trace()
				else:
					new_alpha = 0
					break

			print 'magnitude : ' , new_mag

			if type(last_W) != type(None):
				relative_magnitude = np.linalg.norm(new_W)
				distance_change_since_last_update = np.linalg.norm(last_W - new_W)

				print "\t\t\texit condition : " , distance_change_since_last_update/relative_magnitude
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

	db['W_matrix'] = db['W_matrix'][:,0:db['q']]
	#print db['W_matrix']

#def W_optimize_Gaussian(db):
#	y_tilde = create_y_tilde(db)
#	db['y_tilde'] = y_tilde
#	
#	previous_gw = np.ones((db['N'],db['N']))
#	w_converged = False
#	last_W = None
#
#	for m in range(db['q']):
#		db['W_matrix'][:,m] = get_orthogonal_vector(db, m, db['W_matrix'][:,m])
#		counter = 0
#		new_alpha = 1
#		print 'Starting a new dimension'
#		while not w_converged:
#			w_l = db['W_matrix'][:,m]
#
#			i_values = np.random.permutation( np.array(range(db['N'])) )
#			i_values = i_values[0:db['N']] #SGD_size
#			j_values = np.random.permutation( np.array(range(db['N'])) )
#			j_values = j_values[0:db['N']]
#
#			[update_direction, db['updated_magnitude']] = Stochastic_W_gradient(db, y_tilde, previous_gw, w_l, i_values, j_values)
#			update_direction = get_orthogonal_vector(db, m+1, update_direction) # m+1 is to also remove the current dimension
#
#			#new_alpha = 1	# this is with back trace
#			new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
#			#import pdb; pdb.set_trace()
#			
#			[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)
#
#			print 'Previous mag : ', db['updated_magnitude'], 'New mag : ' , new_mag, new_alpha
#			while new_mag < db['updated_magnitude']:
#				new_alpha = new_alpha * 0.4
#				#print 'Alpha : ', new_alpha
#				if new_alpha > 0.00001 :
#					new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
#					[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)
#					#import pdb; pdb.set_trace()
#				else:
#					new_alpha = 0
#					break
#
#			print 'magnitude : ' , new_mag
#
#			if type(last_W) != type(None):
#				relative_magnitude = np.linalg.norm(new_W)
#				distance_change_since_last_update = np.linalg.norm(last_W - new_W)
#
#				print "\t\t\texit condition : " , distance_change_since_last_update/relative_magnitude
#				if (distance_change_since_last_update/relative_magnitude) < 0.001 : 
#					w_converged = True
#					try:
#						db.pop('previous_wolfe_direction')
#						db.pop('previous_wolfe_magnitude')
#					except: pass
#
#			#print new_W
#			last_W = new_W/np.linalg.norm(new_W)
#			db['W_matrix'][:,m] = last_W
#			counter += 1
#			if counter > db['maximum_W_update_count']: 
#				print '\nExit due to maximum update reached'
#				w_converged = True
#				try:
#					db.pop('previous_wolfe_direction')
#					db.pop('previous_wolfe_magnitude')
#				except: pass
#
#		w_converged = False
#		previous_gw = update_previous_gw(db, last_W, previous_gw)
#
#	db['W_matrix'] = db['W_matrix'][:,0:db['q']]
#	#print db['W_matrix']
