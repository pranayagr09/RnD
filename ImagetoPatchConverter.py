import cv2
import os
import numpy as np 
import glob
from numpy.random import randint

imageDir = "../TrainImages/"
files = [file for file in glob.glob(imageDir+"*.jpg")]
images = [cv2.imread(file) for file in glob.glob(imageDir+"*.jpg")]

patches = []
patches_per_image = 40
for im in images:
	shape = np.shape(im)
	im_height = shape[0]
	im_width =  shape[1]
	rows = randint(0, im_height-96, patches_per_image)
	cols = randint(0, im_width-96, patches_per_image)
	for i in range(0,patches_per_image):
		patches.append(im[rows[i]:rows[i]+96,cols[i]:cols[i]+96,:])
	 
write_path = "../TrainImagePatches/"
for i in range(0,len(patches)):
	patch = patches[i]
	cv2.imwrite(os.path.join(write_path , str(i+1)+'.jpg'), patch)