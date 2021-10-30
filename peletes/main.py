import os
import cv2
import sys
import struct
import copy
import numpy as np
from scipy import  signal
import matplotlib.pyplot as plt



class processPeletes:

    def avg(img, kernel=(5,5)):
        return cv2.blur(img, kernel)
    
    def median(img,kernelSize=3):
         return cv2.medianBlur(img, kernelSize)

    def sobel(img, flagGaussianFilter=False):
        #edge detection using sobel operator
        # converting to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove noise
        if flagGaussianFilter:
            img = processPeletes.gaussian(img)

        # convolute with proper kernels
        #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)  # y
        return sobely

    def laplacian(img, flagGaussianFilter=False):
        #edge detection using sobel operator
        # converting to gray scale
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove noise
        if flagGaussianFilter:
            img = processPeletes.gaussian(img)

        # convolute with proper kernels
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        return laplacian

    def gaussian(img):
        # remove noise
        img = cv2.GaussianBlur(img,(3,3),0)
        return img
    


    


    def main():
        neuromorphicImage = cv2.imread('img/2.jpg')
        
        deepCopy = copy.deepcopy(neuromorphicImage)
        img = deepCopy
        img = processPeletes.sobel(img)
        resized = cv2.resize(img, (int(img.shape[1] * 0.2),int(img.shape[0] * 0.2)), interpolation = cv2.INTER_AREA)
        cv2.imshow('output',resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
       
if __name__ == "__main__":
    processPeletes.main()