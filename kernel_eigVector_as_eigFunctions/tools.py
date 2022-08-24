
import numpy as np
from IPython.display import clear_output, HTML, Math
import pandas as pd




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


