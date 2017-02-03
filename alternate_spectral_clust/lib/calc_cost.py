
from HSIC import *
#	This is deprecated function

def calc_cost_function(db):
	W = db['W_matrix']
	X = db['data']
	XW = X.dot(W)
	U = db['U_matrix']
	Y = db['Y_matrix'][:,0:db['C_num']]
	sigma = db['sigma']
	l = db['lambda']

	cost = l*HSIC_rbf(XW, Y, sigma, Y_kernel='linear') - HSIC_rbf(XW, U, sigma, Y_kernel='linear') 

	#print 'Changed'
	#print l*HSIC_rbf(XW, Y, sigma, Y_kernel='linear')
	#print HSIC_rbf(XW, U, sigma, Y_kernel='linear') 

	return cost



def cost_2(db, W):
	X = db['data']
	XW = X.dot(W)
	U = db['U_matrix']
	Y = db['Y_matrix']
	sigma = db['sigma']
	l = db['lambda']

	cost = l*HSIC_rbf(XW, Y, sigma, Y_kernel='linear') - HSIC_rbf(XW, U, sigma, Y_kernel='linear') 

	print 'Changed'
	print l*HSIC_rbf(XW, Y, sigma, Y_kernel='linear')
	print HSIC_rbf(XW, U, sigma, Y_kernel='linear') 

	return cost

