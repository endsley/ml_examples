#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio


#x = np.array([[2,3,1,0],[0,0,3,4]])
#print x
#s = np.transpose(np.nonzero(x))
##s = np.count_nonzero(x)
#print s





img = cv2.imread('yoga_3.png',0)
#edges = cv2.Canny(img,100,200)
edges = cv2.Canny(img,50,100)
s = np.transpose(np.nonzero(edges))
sio.savemat('points_3a.mat', {'points_3a':s})


img = cv2.imread('yoga_3_b.png',0)
#edges = cv2.Canny(img,100,200)
edges = cv2.Canny(img,50,100)
s = np.transpose(np.nonzero(edges))
sio.savemat('points_3b.mat', {'points_3b':s})


img = cv2.imread('yoga_3_c.png',0)
#edges = cv2.Canny(img,100,200)
edges = cv2.Canny(img,50,100)
s = np.transpose(np.nonzero(edges))
sio.savemat('points_3c.mat', {'points_3c':s})










#print s
#
#print edges.size
#i = edges.shape[0]
#j = edges.shape[1]
#
#
#for m in 
#	print edges[60][m], 

##cv2.imwrite( "./posture_outline.jpg", edges) 
#
#
##plt.subplot(121),plt.imshow(img,cmap = 'gray')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##plt.subplot(122),plt.imshow(edges,cmap = 'gray')
##plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
##
##plt.show()
