import  numpy as np
import math
import cv2 as opencv

img = opencv.imread('/home/pranay/Desktop/RnD/flownet/Pranay/TrainImagePatches/136.jpg')
R, C = np.meshgrid(np.arange(96)-48,np.arange(96)-48)

FT = np.zeros((96,96), dtype=complex)

A = np.random.rand(96,96)
phase = np.random.rand(96,96)*math.pi*2

FT = A*np.exp(-1j*phase)

mask = (R**2 + C**2)< 8**2
# mask = np.logical_and((R**2 + C**2) < 8**2,np.logical_and(R<0,C<0))
FT *= mask
FT = np.fft.ifftshift(FT)

of = np.fft.ifft2(FT).real


of = 8*((of-of.min())/(of.max()-of.min()) - .5)

ofu = of; ofv = of

mapR = C + ofu + 48
mapC = R + ofv + 48

mapR[mapR<0]=0; mapR[mapR>95] = 95
mapC[mapC<0]=0; mapC[mapC>95] = 95

warpedImg = img[mapR.astype(int),mapC.astype(int),:]
opencv.imshow('',warpedImg); opencv.waitKey(0)


