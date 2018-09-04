from scipy import interpolate
import numpy as np
import math
import random

class PatchDistorter:

	def getRandDistortion(self,patch):
		patch_size =  np.shape(patch)[0]

		x = np.arange(patch_size)
		y = np.arange(patch_size)
			
		Fr = interpolate.interp2d(x, y, patch[:,:,0])
		Fg = interpolate.interp2d(x, y, patch[:,:,1])
		Fb = interpolate.interp2d	(x, y, patch[:,:,2])

		R, C = np.meshgrid(np.arange(patch_size)-patch_size/2,np.arange(patch_size)-patch_size/2)
		mask_size = 8
		# mask = (R**2 + C**2) < mask_size**2 && R<0 && C<0
		mask = np.logical_and((R**2 + C**2) < mask_size**2,np.logical_and(R<0,C<0))

		of_u1 = self.getOf(patch_size, mask)
		of_v1 = self.getOf(patch_size, mask)

		forward_map_R1 = R[16:80,16:80] + of_u1[16:80,16:80] + patch_size/2
		# forward_map_R1 = R + of_u1 + patch_size/2
		forward_map_C1 = C[16:80,16:80] + of_v1[16:80,16:80] + patch_size/2
		# forward_map_C1 = C + of_v1 + patch_size/2
		# forward_map_R1[forward_map_R1<0]=0; forward_map_R1[forward_map_R1>95] = 95
		# forward_map_C1[forward_map_C1<0]=0; forward_map_C1[forward_map_C1>95] = 95
		warpedPatch1 = patch[forward_map_C1.astype(int),forward_map_R1.astype(int),:]

		# R1 = forward_map_R1.flatten()
		# C1 = forward_map_C1.flatten()
		# warpedPatch = np.array((Fr(R1, C1), Fg(R1,C1), Fb(R1, C1)), dtype = np.uint8)
		# return np.shape(of_u1)
		# warpedPatch1 = np.reshape(warpedPatch, (96,96,3))
		# warpedPatch1 = np.array((Fr(forward_map_R1, forward_map_C1), Fg(forward_map_R1,forward_map_C1), Fc(forward_map_R1, forward_map_R1)), dtype = np.uint8)

		of_u2 = self.getOf(patch_size, mask)
		of_v2 = self.getOf(patch_size, mask)
		forward_map_R2 = R[16:80,16:80] + of_u2[16:80,16:80] + patch_size/2
		# forward_map_R2 = R + of_u2 + patch_size/2
		forward_map_C2 = C[16:80,16:80] + of_v2[16:80,16:80] + patch_size/2
		# forward_map_C2 = C + of_v2 + patch_size/2
		# forward_map_R1[forward_map_R1<0]=0; forward_map_R1[forward_map_R1>95] = 95
		# forward_map_C1[forward_map_C1<0]=0; forward_map_C1[forward_map_C1>95] = 95
		warpedPatch2 = patch[forward_map_C2.astype(int),forward_map_R2.astype(int),:]

		of_u = of_u2 - of_u1
		of_v = of_v2 - of_v1
		of = np.sqrt((of_u**2 + of_v**2))
		warpedOf = of[forward_map_C1.astype(int),forward_map_R1.astype(int)]

		return warpedPatch1, warpedPatch2, warpedOf


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