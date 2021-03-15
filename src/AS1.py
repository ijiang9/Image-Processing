# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import math
import myFunction 

#'S' smoothing using my convolution function
def sliderHandler1(x):
    global src
    global img_p
    if (x%2==0):
        x = x+1
    kernel = np.ones((x,x),np.float32)/(x*x)  
    #using my convolution function    
    img_p = myFunction.myConvolve(src, kernel)
    cv2.imshow(winName, img_p)
#'s' smoothing using built-in function
def sliderHandler2(x):
    global src
    global img_p
    if (x%2==0):
        x = x+1
    img_p = cv2.medianBlur(src,x)
    cv2.imshow(winName, img_p)
#'p' plot gradient vectors
def sliderHandler3(x):
    global img
    global img_p
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_p = np.ones((img.shape[0],img.shape[1],3),np.uint8)
    img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=5)
    for i in range(0, src.shape[0],x):
        for j in range(0,src.shape[1],x):
            angle = math.atan2(sobelx[i,j],sobely[i,j])
            endX = int(i + x * math.cos(angle))
            endY = int(j + x * math.sin(angle))
            cv2.arrowedLine(img_p, (j,i), (endY,endX),(0,0,0))
    cv2.imshow(winName, img_p)
#'r' rotate image
def sliderHandler4(x):
    global img
    global img_p
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_p = np.ones((img.shape[0],img.shape[1],3),np.uint8)
    img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    angle = x*np.pi/180
    rows = src.shape[0]
    cols = src.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2),x,1)
    img_p = cv2.warpAffine(src, M, (cols, rows))
    cv2.imshow(winName, img_p)

def helper():
    
    print ("i=reload image")
    print ("w=save current processed image")
    print ("g=convert to grayscale")
    print ("G=convert to grayscale using my function")
    print ("c=cycle through color channels")
    print ("s=smoothing")
    print ("S=smoothing usiong my convolution function")
    print ("d=downsample by a factor of 2 without smoothing")
    print ("D=downsample by a factor of 2 with smoothing")
    print ("x=x derivative filter")
    print ("y=y derivative filter")
    print ("m=magnitude of gradient")
    print ("p=plot gradient vector")
    print ("r=rotate image")

winName = 'Image Processing'
channel = 0
img = cv2.imread('images.jpeg')
#img = cv2.resize(img,(800,600))
while img.shape[0] > 1200 or img.shape[1] > 750:
    img = cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow(winName, img)
img_p = img

while(True):
    key = cv2.waitKey(10) & 0xFF
    
    #read image file
    if key == ord('i'): 
        img = cv2.imread('images.jpeg')
#        img = cv2.resize(img,(800,600))
        while img.shape[0] > 1200 or img.shape[1] > 750:
            img = cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)))
        cv2.imshow(winName, img)
        img_p = img
    #save processed image   
    elif key == ord('w'):
        cv2.imwrite('out.jpg',img_p)
    #convert to grayscale
    elif key == ord('g'):
    
        img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow(winName,img_p)
    #convert to grayscale using my convolution func
    elif key & 255 == ord('G'):
        print(key)
        img_p = np.ones((img.shape[0],img.shape[1],3),np.uint8)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                img_p[i,j] = img[i,j,0] * 0.299 + img[i,j,1] * 0.587 + img[i,j,2] * 0.114
        cv2.imshow(winName, img_p)
    #cycle through color channels
    elif key == ord('c'):
        if channel == 0:
            img_p = img.copy()
            img_p[:,:,1] = 0
            img_p[:,:,2] = 0
            channel = 1
            
        elif channel == 1:
            img_p = img.copy()
            img_p[:,:,0] = 0
            img_p[:,:,2] = 0
            channel = 2
        else:
            img_p = img.copy()
            img_p[:,:,0] = 0
            img_p[:,:,1] = 0
            channel = 0
        cv2.imshow(winName,img_p)
    #smoothing using my covolution func
    elif key == ord('S'):
        
        img_p = np.ones((img.shape[0],img.shape[1],3),np.uint8)
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.createTrackbar('mySmoothing',winName,1,30,sliderHandler1)
    #smoothing   
    elif key == ord('s'):
        
        img_p = np.ones((img.shape[0],img.shape[1],3),np.uint8)
        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.createTrackbar('smoothing',winName,1,30,sliderHandler2)
    #downsampling without smoothing
    elif key == ord('d'):
        img_p = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        cv2.imshow(winName, img_p)
    #downsampling with smoothing    
    elif key == ord('D'):
        img_p = cv2.pyrDown(img)
        cv2.imshow(winName, img_p)
    #x derivative filter
    elif key == ord('x'):
        img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(img_p,cv2.CV_64F,1,0,ksize=5)
#        sobelXnorm = sobelx/sobelx.max()*255
#        sobelx = np.uint8(sobelXnorm)
        img_p = cv2.normalize(sobelx, img_p, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
#        img_p = sobelx
        cv2.imshow(winName, img_p)
    #y derivative filter
    elif key == ord('y'):
        img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(img_p,cv2.CV_64F,0,1,ksize=5)
#        sobelynorm = sobely/sobely.max()*255
#        sobely = np.uint8(sobelynorm)
#        img_p = sobely
        img_p = cv2.normalize(sobely, img_p, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        cv2.imshow(winName, img_p)
    #magnitude of gradient
    elif key == ord('m'):
        img_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(img_p,cv2.CV_64F,1,1,ksize=5)
#        sobelnorm = sobel/sobel.max()*255
#        sobel = np.uint8(sobelnorm)
#        img_p = sobel
        img_p = cv2.normalize(sobel, img_p, alpha = 0, beta = 1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        cv2.imshow(winName, img_p)
    #plot gradient vector
    elif key == ord('p'):
        cv2.createTrackbar('Num of Pixels',winName,0,100,sliderHandler3)
    #rotate image
    elif key == ord('r'):
        cv2.createTrackbar('Angle',winName,0,360,sliderHandler4)
    #helper func
    elif key == ord('h'):
        helper()
    #exit
    elif key == 27:
        cv2.destroyWindow(winName)
        break
    else:
        pass

