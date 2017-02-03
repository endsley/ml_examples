
def append_txt(filename, append_txt):
	fin = open(filename, 'a')
	fin.write(append_txt)
	fin.close()
