
import numpy as np

def Y_2_allocation(Y):
	i = 0
	allocation = np.array([])
	for m in range(Y.shape[0]):
		allocation = np.hstack((allocation, np.where(Y[m] == 1)[0][0]))
		i += 1

	return allocation

def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,4))
	Y = np.delete(Y, 0, 0)

	return Y
