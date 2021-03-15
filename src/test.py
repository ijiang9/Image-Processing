#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:25:03 2020

@author: yinjiang
"""
import cv2
import myFunction
import numpy as np
import time

def myConvolve(img, kernel):
    kx, ky = kernel.shape
    ix, iy = img.shape
    img_p = np.ones((ix,iy),np.uint8)
    pad = int((kx - 1) / 2)
    
    imagePad = cv2.copyMakeBorder(img, pad, pad, pad, pad,
		cv2.BORDER_CONSTANT)
    
    for i in range(pad,ix + pad):
        for j in range(pad,iy + pad):
            window = imagePad[i - pad:i + pad+1, j - pad:j + pad+1]
            img_p[i-pad,j-pad] = (window * kernel).sum()
    img_p = (img_p - img_p.min())/(img_p.max() - img_p.min())
    
    img_p = (img_p * 255).astype("uint8")
    return img_p


img = cv2.imread('images.jpeg')
kernel = np.ones((5,5),np.float32)/(25)  
src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
start_time = time.time()
myFunction.myConvolve(src, kernel)
print("--- Cython Function: %s seconds ---" % (time.time()-start_time))
start_time = time.time()
myConvolve(src, kernel)
print("--- Python Function: %s seconds ---" % (time.time()-start_time))