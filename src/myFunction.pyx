#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:04:34 2020

@author: yinjiang
"""
import cv2
import numpy as np
def myConvolve(img, kernel):
    cdef int kx, ky, ix, iy
    kx, ky = kernel.shape
    ix, iy = img.shape
    img_p = np.ones((ix,iy),np.uint8)
    pad = (kx - 1) / 2
    
    imagePad = cv2.copyMakeBorder(img, pad, pad, pad, pad,
		cv2.BORDER_CONSTANT)
    
    for i in range(pad,ix + pad):
        for j in range(pad,iy + pad):
            window = imagePad[i - pad:i + pad+1, j - pad:j + pad+1]
            img_p[i-pad,j-pad] = (window * kernel).sum()
    img_p = (img_p - img_p.min())/(img_p.max() - img_p.min())
    
    img_p = (img_p * 255).astype("uint8")
    return img_p