
import numpy as np
import sklearn
from numpy import genfromtxt
from sklearn.utils import shuffle
from IPython.display import clear_output, HTML, Math
import pandas as pd
import sys

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

def mean_absolute_error(U1, U2, scale):
	U1 = np.absolute(U1)
	U2 = np.absolute(U2)

	return np.sum(np.absolute(U1 - U2))/scale

def isnotebook():
	try:
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell':
			return True   # Jupyter notebook or qtconsole
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False	  # Probably standard Python interpreter


def wtype(data):
	return type(data).__name__



def pretty_np_array(m, front_tab='', verticalize=False, title=None, auto_print=False, end_space=''):
	m = str(m)

	if verticalize:
		if len(m.shape) == 1:
			m = np.atleast_2d(m).T

	out_str = front_tab + str(m).replace('\n ','\n' + front_tab).replace('[[','[').replace(']]',']') + end_space + '\n'
	out_str = str(out_str).replace('.]',']')

	if type(title).__name__ == 'str':
		L1 = out_str.split('\n')
		L1_max_width = len(max(L1, key=len))
		t1 = str.center(title, L1_max_width)
		out_str = t1 + '\n' + out_str

	if auto_print: print(out_str)
	else: return out_str

def block_two_string_concatenate(str1, str2, spacing='\t', add_titles=[], auto_print=False):
	str1 = str(str1)
	str2 = str(str2)

	L1 = str1.split('\n')
	L2 = str2.strip().split('\n')

	if len(L1) > len(L2):
		Δ = len(L1) - len(L2)
		for ι in range(Δ):
			L2.append('\n')

	if len(add_titles) == 2:
		L1_max_width = len(max(L1, key=len))
		L2_max_width = len(max(L2, key=len))
		t1 = str.center(add_titles[0], L1_max_width)
		t2 = str.center(add_titles[1], L2_max_width)
		L1.insert(0,t1)
		L2.insert(0,t2)

	max_width = len(max(L1, key=len))
	outS = ''
	for l1, l2 in zip(L1,L2):
		outS += ('%-' + str(max_width) + 's' + spacing + l2 + '\n') % l1

	if auto_print: print(outS)
	else: return outS


def print_two_matrices_side_by_side(M1, M2, title1='', title2='', auto_print=True):
	eK = pretty_np_array(M1, front_tab='', title=title1, auto_print=False)
	eQ = pretty_np_array(M2, front_tab='', title=title2, auto_print=False)
	block_two_string_concatenate(eK, eQ, spacing='\t', add_titles=[], auto_print=auto_print)

def get_rbf_γ(X):
	σ = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
	γ = 1.0/(2*σ*σ)
	return γ


def csv_load(xpath, ypath=None, shuffle_samples=False):
	X = genfromtxt(	xpath, delimiter=',')
	Y = None
	if ypath is not None: 
		Y = genfromtxt(ypath, delimiter=',')

	if shuffle_samples:
		if Y is None:
			X = shuffle(X, random_state=0)
		else:
			X, Y = shuffle(X, Y, random_state=0)

	if Y is None: return X
	else: return [X,Y]


def jupyter_print(value, display_all_rows=False, display_all_columns=False, font_size=3, latex=False):
	#font_size is from 1 to 6
	font_size = int(6 - font_size)
	if font_size > 6: font_size = 6
	if font_size < 0: font_size = 1

	if isnotebook():
		if wtype(value) == 'DataFrame': 
			if display_all_rows: pd.set_option('display.max_rows', None)
			if display_all_columns: pd.set_option('display.max_columns', None) 
			display(value)
		elif wtype(value) == 'wData': 
			if display_all_rows: pd.set_option('display.max_rows', None)
			if display_all_columns: pd.set_option('display.max_columns', None)
			display(value.df)
		elif wtype(value) == 'Index': 
			value = str(value.tolist())
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wtype(value) == 'tuple': 
			value = str(value)
			str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
			display(HTML(data=str2html))
		elif wtype(value) == 'str': 
			value = value.replace('\r','<br>')
			value = value.replace('\n','<br>')
			if latex:
				display(Math(r'%s'%value))
			else:
				str2html = '<html><body><h%d>%s</h%d></body></html>'%(font_size, value, font_size)
				display(HTML(data=str2html))
		else:
			print(value)

		pd.set_option('display.max_rows', 10)
	else:
		print(value)


