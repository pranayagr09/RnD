from scipy import interpolate
import numpy as np
import math
import random
import cv2
import glob

class PatchDistorter:

	def getRandDistortion(self,patch):
		patch_size =  np.shape(patch)[0]

		x = np.arange(patch_size)
		y = np.arange(patch_size)
		points_xy = np.transpose([np.repeat(y, len(x)), np.tile(x, len(y))])	

		patch_R = patch[:,:,0].flatten()
		patch_G = patch[:,:,1].flatten()
		patch_B = patch[:,:,2].flatten()

		R, C = np.meshgrid(np.arange(patch_size)-patch_size/2,np.arange(patch_size)-patch_size/2)
		mask_size = 5
		mask = np.logical_and((R**2 + C**2) < mask_size**2,np.logical_and(R<0,C<0))

		of_u1 = self.getOf(patch_size, mask)
		of_v1 = self.getOf(patch_size, mask)

		reverse_map_R1 = R[16:80,16:80] + of_u1[16:80,16:80] + patch_size/2
		reverse_map_C1 = C[16:80,16:80] + of_v1[16:80,16:80] + patch_size/2
		# reverse_map_R1 = R + of_u1 + patch_size/2
		# reverse_map_C1 = C + of_v1 + patch_size/2
		# reverse_map_R1[reverse_map_R1<0]=0; reverse_map_R1[reverse_map_R1>95] = 95
		# reverse_map_C1[reverse_map_C1<0]=0; reverse_map_C1[reverse_map_C1>95] = 95
		warpedPatch = patch[reverse_map_C1.astype(int),reverse_map_R1.astype(int),:]

		warpedPatch_R = interpolate.griddata(points_xy, patch_R, (reverse_map_C1,reverse_map_R1), method = 'linear')
		warpedPatch_G = interpolate.griddata(points_xy, patch_G, (reverse_map_C1,reverse_map_R1), method = 'linear')
		warpedPatch_B = interpolate.griddata(points_xy, patch_B, (reverse_map_C1,reverse_map_R1), method = 'linear')

		warpedPatch_R[warpedPatch_R<0.0] = 0.0; warpedPatch_R[warpedPatch_R>255.0] = 255.0;
		warpedPatch_G[warpedPatch_G<0.0] = 0.0; warpedPatch_G[warpedPatch_G>255.0] = 255.0;
		warpedPatch_B[warpedPatch_B<0.0] = 0.0; warpedPatch_B[warpedPatch_B>255.0] = 255.0;		

		warpedPatch1 = np.dstack((np.dstack((warpedPatch_R, warpedPatch_G)),warpedPatch_B))
		warpedPatch1 = cv2.normalize(warpedPatch1, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
		# print(warpedPatch1)
		# print(warpedPatch1-warpedPatch2)


		of_u2 = self.getOf(patch_size, mask)
		of_v2 = self.getOf(patch_size, mask)
		
		reverse_map_R = R + of_u2 + patch_size/2
		reverse_map_C = C + of_v2 + patch_size/2
		ofu1 = of_u1.flatten()
		ofv1 = of_v1.flatten()

		interp_ofu1 = interpolate.griddata(points_xy, ofu1, (reverse_map_C,reverse_map_R), method = 'linear')
		interp_ofv1 = interpolate.griddata(points_xy, ofv1, (reverse_map_C,reverse_map_R), method = 'linear')

		reverse_map_R2 = R[16:80,16:80] + interp_ofu1[16:80,16:80] + patch_size/2
		reverse_map_C2 = C[16:80,16:80] + interp_ofv1[16:80,16:80] + patch_size/2

		# reverse_map_R2 = R[16:80,16:80] + of_u2[16:80,16:80] + patch_size/2
		# reverse_map_C2 = C[16:80,16:80] + of_v2[16:80,16:80] + patch_size/2

		warpedPatch_R = interpolate.griddata(points_xy, patch_R, (reverse_map_C2,reverse_map_R2), method = 'linear')
		warpedPatch_G = interpolate.griddata(points_xy, patch_G, (reverse_map_C2,reverse_map_R2), method = 'linear')
		warpedPatch_B = interpolate.griddata(points_xy, patch_B, (reverse_map_C2,reverse_map_R2), method = 'linear')

		warpedPatch_R[warpedPatch_R<0.0] = 0.0; warpedPatch_R[warpedPatch_R>255.0] = 255.0;
		warpedPatch_G[warpedPatch_G<0.0] = 0.0; warpedPatch_G[warpedPatch_G>255.0] = 255.0;
		warpedPatch_B[warpedPatch_B<0.0] = 0.0; warpedPatch_B[warpedPatch_B>255.0] = 255.0;		

		warpedPatch2 = np.dstack((np.dstack((warpedPatch_R, warpedPatch_G)),warpedPatch_B))
		warpedPatch2 = cv2.normalize(warpedPatch2, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

		# cv2.imshow('', warpedPatch1-warpedPatch2)
		# cv2.waitKey(0)

		return warpedPatch1, warpedPatch2
		# return warpedPatch1, warpedPatch2, warpedOf


	def getOf(self,patch_size, mask):
		max_pixel_shift = 4
		
		fft = np.zeros((patch_size,patch_size), dtype = complex)
		mask_cells = np.sum(mask) 
		amplitude = np.random.normal(0,0.25,mask_cells)
		phase = np.random.rand(mask_cells)*math.pi*2
		fft[mask] = amplitude*np.exp(-1j*phase)

		fft = np.fft.ifftshift(fft)
		of = np.fft.ifft2(fft).real
		of = 2*max_pixel_shift*((of-of.min())/(of.max()-of.min()) - .5)		
		return of


imageDir = "../TrainImagePatches/"
files = [file for file in glob.glob(imageDir+"*.jpg")]
imagePatches = [cv2.imread(file) for file in glob.glob(imageDir+"*.jpg")]
testPatches = random.sample(imagePatches,50)
Distorter = PatchDistorter()

test_patch = testPatches[2]
cv2.imshow('patch', test_patch)
cv2.waitKey(0)
w1,w2 = Distorter.getRandDistortion(test_patch)
cv2.imshow('w1', w1)
cv2.waitKey(0)
cv2.imshow('w2', w2)
cv2.waitKey(0)