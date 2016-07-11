
import numpy as np
from create_A_ij_matrix import *
from create_gamma_ij import *
from create_y_tilde import *
from objective_magnitude import *


def create_A_ij_matrix(db, i, j):
	x_i = db['data'][i][np.newaxis]
	x_j = db['data'][j][np.newaxis]
	return x_i.T.dot(x_j)


def Stochastic_W_gradient(db, y_tilde, previous_gw, w_l, i_values, j_values):
	update_direction = 0
	updated_magnitude = 0

	for i in i_values:
		for j in j_values:
			#import pdb; pdb.set_trace()
			A_ij = create_A_ij_matrix(db,i,j)
			gamma_ij = create_gamma_ij(db, y_tilde, i, j)

			wAw = w_l.T.dot(A_ij).dot(w_l) + previous_gw[i,j] + 1
			front_constant_term = gamma_ij*db['poly_order']
			polynomial_term = np.power(wAw, db['poly_order'] - 1)
			derivative_term = (A_ij + A_ij.T).dot(w_l)


			portion_magnitude = gamma_ij*np.power(wAw, db['poly_order'])
			updated_magnitude += portion_magnitude
			update_direction += front_constant_term*polynomial_term*derivative_term

	
	#print 'updated_magnitud : ' , updated_magnitude
	return [update_direction, updated_magnitude]

def update_previous_gw(db, w, previous_gw):
	for i in range(db['N']):
		for j in range(db['N']):
			A_ij = create_A_ij_matrix(db,i,j)

			new_term = w.T.dot(A_ij).dot(w) + previous_gw[i,j]
			previous_gw[i,j] = previous_gw[i,j] + new_term

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


#	This will implement the mini-batch Gradient Descent
def W_optimize_polynomial(db):
	y_tilde = create_y_tilde(db)
	#print 'y tilde \n', y_tilde
	
	previous_gw = np.zeros((db['N'],db['N']))
	w_converged = False
	last_W = None

	for m in range(db['q']):
		db['W_matrix'][:,m] = get_orthogonal_vector(db, m, db['W_matrix'][:,m])
		counter = 0
		new_alpha = 0.5

		while not w_converged:
			w_l = db['W_matrix'][:,m]

			i_values = np.random.permutation( np.array(range(db['N'])) )
			i_values = i_values[0:db['SGD_size']]
			j_values = np.random.permutation( np.array(range(db['N'])) )
			j_values = j_values[0:db['SGD_size']]


			#i_values = [0,1]
			#j_values = [0,1]


			[update_direction, db['updated_magnitude']] = Stochastic_W_gradient(db, y_tilde, previous_gw, w_l, i_values, j_values)
			update_direction = get_orthogonal_vector(db, m+1, update_direction) # m+1 is to also remove the current dimension

			new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction	
			[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)

			#om = objective_magnitude
			#print m, db['updated_magnitude'], new_mag, w_l, new_alpha
			#import pdb; pdb.set_trace()

			while new_mag < db['updated_magnitude']:
				new_alpha = new_alpha * 0.8
				if new_alpha > 0.0001 :
					new_W = np.sqrt(1-new_alpha*new_alpha)*w_l + new_alpha*update_direction
					[tmp_dir, new_mag] = Stochastic_W_gradient(db, y_tilde, previous_gw, new_W, i_values, j_values)
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

	db['W_matrix'] = db['W_matrix'][:,0:db['q']]
	#print db['W_matrix']
