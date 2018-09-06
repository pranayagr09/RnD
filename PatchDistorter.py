from scipy import interpolate
import numpy as np
import math
import random
import cv2
import glob
import time

class PatchDistorter:

	def getInterpolatedImage(self, image, mappingR, mappingC):
		T = np.floor(mappingR).astype(int)
		L = np.floor(mappingC).astype(int)
		R = L + 1
		B = T + 1
		R = R.clip(0, image.shape[0]-1)
		B = B.clip(0, image.shape[1]-1)

		wtTL = (B - mappingR) * (R - mappingC)
		wtTR = (B - mappingR) * (mappingC - L)
		wtBR = (mappingR - T) * (mappingC - L)
		wtBL = (mappingR - T) * (R - mappingC)


		imgTL =  image[T, L]
		imgTR =  image[T, R]
		imgBR =  image[B, R]
		imgBL =  image[B, L]

		interpImg = imgTL * wtTL[:, :, None] + imgTR * wtTR[:, :, None] + imgBR * wtBR[:, :, None] + imgBL * wtBL[:, :, None]
		return interpImg

	def getRandDistortion(self,patch):
		patch_size =  np.shape(patch)[0]

		x = np.arange(patch_size)
		y = np.arange(patch_size)
		R, C = np.meshgrid(np.arange(patch_size)-patch_size/2,np.arange(patch_size)-patch_size/2)
		mask_size = 5
		mask = np.logical_and((R**2 + C**2) < mask_size**2,np.logical_and(R<0,C<0))

		R = R + patch_size/2
		C = C + patch_size/2

		of_u1 = self.getOf(patch_size, mask, 4)
		of_v1 = self.getOf(patch_size, mask, 4)

		reverse_map_R1 = R + of_u1
		reverse_map_C1 = C + of_v1 
		reverse_map_1 = np.dstack((reverse_map_R1, reverse_map_C1))

		warpedPatch1 = self.getInterpolatedImage(patch, reverse_map_C1[16:80,16:80], reverse_map_R1[16:80,16:80])/255.0

		of_u2 = self.getOf(patch_size, mask, 3)
		of_v2 = self.getOf(patch_size, mask, 3)
		
		reverse_map_R = R + of_u2 
		reverse_map_C = C + of_v2 
		reverse_map_R[reverse_map_R<0]=0; reverse_map_R[reverse_map_R>95] = 95
		reverse_map_C[reverse_map_C<0]=0; reverse_map_C[reverse_map_C>95] = 95

		interp_RM2 = self.getInterpolatedImage(reverse_map_1, reverse_map_C, reverse_map_R)
		reverse_map_R2 = interp_RM2[:,:,0] 
		reverse_map_C2 = interp_RM2[:,:,1]
		# reverse_map_R2[reverse_map_R2<0]=0; reverse_map_R2[reverse_map_R2>95] = 95
		# reverse_map_C2[reverse_map_C2<0]=0; reverse_map_C2[reverse_map_C2>95] = 95 
		
		warpedPatch2 = self.getInterpolatedImage(patch, reverse_map_C2[16:80,16:80], reverse_map_R2[16:80,16:80])/255.0

		# print(np.mean(np.abs(reverse_map_1)))
		# print(np.mean(np.abs(interp_RM2)))
		# print(np.mean(np.abs(interp_RM2 - reverse_map_1)))


		# reverse_map_R3 = R + of_u2
		# reverse_map_C3 = C + of_v2
		# reverse_map_R3[reverse_map_R3<16]=16; reverse_map_R3[reverse_map_R3>79] = 79
		# reverse_map_C3[reverse_map_C3<16]=16; reverse_map_C3[reverse_map_C3>79] = 79

		# warpedPatch3 = self.getInterpolatedImage(warpedPatch1, reverse_map_C3[16:80,16:80] -16, reverse_map_R3[16:80,16:80] - 16)

		# print(np.mean(np.abs(warpedPatch3-warpedPatch2)))

		# print(interp_ofv1 - of_v1)
		return warpedPatch1, warpedPatch2
		# return warpedPatch1, warpedPatch2, warpedOf


	def getOf(self,patch_size, mask, max_pixel_shift):
		
		fft = np.zeros((patch_size,patch_size), dtype = complex)
		mask_cells = np.sum(mask) 
		amplitude = np.random.normal(0,0.25,mask_cells)
		phase = np.random.rand(mask_cells)*math.pi*2
		fft[mask] = amplitude*np.exp(-1j*phase)

		fft = np.fft.ifftshift(fft)
		of = np.fft.ifft2(fft).real
		of = ((of-of.min())/(of.max()-of.min()) - .5)	
		of = max_pixel_shift * of 	
		return of


imageDir = "../TrainImagePatches/"
files = [file for file in glob.glob(imageDir+"*.jpg")]
imagePatches = [cv2.imread(file) for file in glob.glob(imageDir+"*.jpg")]
testPatches = random.sample(imagePatches,500)
Distorter = PatchDistorter()
t0 = time.time()

for test_patch in testPatches:
	cv2.imshow('patch', test_patch)
	cv2.waitKey(0)
	w1, w2 = Distorter.getRandDistortion(test_patch)
	# w1,w2= Distorter.getRandDistortion(test_patch)
	cv2.imshow('w1', w1)
	cv2.waitKey(0)
	cv2.imshow('w2', w2)
	cv2.waitKey(0)
	# cv2.imshow('w3', w3)
	# cv2.waitKey(0)


t1= time.time()
print(t1-t0)


