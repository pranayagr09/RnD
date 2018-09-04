import cv2
import numpy as np
import glob
import random
from PatchDistorter import PatchDistorter

imageDir = "../TrainImagePatches/"
files = [file for file in glob.glob(imageDir+"*.jpg")]
imagePatches = [cv2.imread(file) for file in glob.glob(imageDir+"*.jpg")]

testPatches = random.sample(imagePatches,50)
Distorter = PatchDistorter()

for patch in testPatches:
	warpedIm1,warpedIm2, warpedOf = Distorter.getRandDistortion(patch)
	cv2.imshow('patch', patch)
	cv2.waitKey(0)
	cv2.imshow('w1', warpedIm1)
	cv2.waitKey(0)
	cv2.imshow('w2', warpedIm2)
	cv2.waitKey(0)
	cv2.imshow('of', warpedOf)
	cv2.waitKey(0)