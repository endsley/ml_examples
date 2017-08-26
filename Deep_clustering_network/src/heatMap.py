#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from numpy import genfromtxt
from sklearn.cluster import SpectralClustering
 

class heatMap:
	def __init__(self):
		self.plot_font_size = 4
		font = {'family' : 'normal', 'weight' : 'bold', 'size'   : self.plot_font_size}
		matplotlib.rc('font', **font)
		self.cluster_by_id = {}
		self.cluster_by_name = {}

	def sort_kernel(self, kernel, allocation, item_labels=[] ):
		alloc_list = np.unique(allocation)
 		ksize = kernel.shape[0]
		self.rearrangement = []

		sorted_kernel = np.empty((0, ksize))
		for m in alloc_list:
			new_list = np.where(allocation == m)[0].tolist()
			self.cluster_by_id[m] = new_list
			self.rearrangement.extend(new_list)

			if len(item_labels) > 0:
				self.cluster_by_name[m] = []
				for q in new_list:
					self.cluster_by_name[m].append( item_labels[q] )


			f = kernel[allocation == m, :]
			sorted_kernel = np.vstack((sorted_kernel, f))
			
		H_sorted_kernel = np.empty((ksize,0))
		for m in alloc_list:
			f = sorted_kernel[:,allocation == m]
			H_sorted_kernel = np.hstack((H_sorted_kernel, f))

		self.sorted_kernel = H_sorted_kernel


		#	Sort the labels if they exist
		if len(item_labels) > 0:
			self.sorted_labels = []
			for m in self.rearrangement:
				self.sorted_labels.append( item_labels[m] )

		return H_sorted_kernel

	def draw_HeatMap(self, kernel, xlabel=[], ylabel=[], title=''):
		ylabel = list(reversed(ylabel))
		
		kernel = np.flipud(kernel)
		fig, ax = plt.subplots()
		#fig.set_size_inches(13,13)
		heatmap = plt.pcolor(kernel, cmap=matplotlib.cm.Blues, alpha=0.8)

		if len(ylabel) > 0:
			ax.set_yticks(np.arange(kernel.shape[0]) + 0.5, minor=False)
			ax.set_yticklabels(ylabel, rotation='horizontal', minor=False)
		
		if len(xlabel) > 0:
			ax.set_xticks(np.arange(kernel.shape[1]) + 0.5, minor=False)
			ax.set_xticklabels(xlabel, rotation='vertical', minor=False)
	 
		plt.title(title)
		plt.show() 

		return plt

	def save_HeatMap(self, path, kernel, xlabel=[], ylabel=[], title=''):
		#xlabel = list(reversed(xlabel))
		ylabel = list(reversed(ylabel))
	
		kernel = np.flipud(kernel)
		fig, ax = plt.subplots()
		#fig.set_size_inches(13,13)
		heatmap = plt.pcolor(kernel, cmap=matplotlib.cm.Blues, alpha=0.8)

		if len(ylabel) > 0:
			ax.set_yticks(np.arange(kernel.shape[0]) + 0.5, minor=False)
			ax.set_yticklabels(ylabel, rotation='horizontal', minor=False)
		
		if len(xlabel) > 0:
			ax.set_xticks(np.arange(kernel.shape[1]) + 0.5, minor=False)
			ax.set_xticklabels(xlabel, rotation='vertical', minor=False)
	 
		plt.title(title)
		plt.draw()	
		fig.savefig(path, dpi=500)
		return plt


if __name__ == "__main__":
#	dat_4 = genfromtxt('../datasets/data_4.csv', delimiter=',')
#	clf = SpectralClustering(n_clusters=4)
	
#	sigma = 0.3
#	Vgamma = 1/(2*sigma*sigma)
#	dat_4 = genfromtxt('../datasets/moon_400_2.csv', delimiter=',')
#	clf = SpectralClustering(n_clusters=2, gamma=Vgamma)

	sigma = 1
	Vgamma = 1/(2*sigma*sigma)
	dat_4 = genfromtxt('../datasets/breast-cancer.csv', delimiter=',')
	#clf = SpectralClustering(n_clusters=2, gamma=Vgamma)
	clf = SpectralClustering(n_clusters=2, gamma=1.5)





	allocation = clf.fit_predict(dat_4)
	kernel = clf.affinity_matrix_
	label = range(kernel.shape[0])
	hMap = heatMap()
	sorted_kernel = hMap.sort_kernel(kernel, allocation)
	hMap.draw_HeatMap(sorted_kernel, xlabel=label, ylabel=label, title='')
