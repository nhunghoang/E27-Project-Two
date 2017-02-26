# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:06:19 2017

@author: Dan
"""
import sys
import numpy as np
import cv2

def lopass(img, sigma, ksize):
    lofil = cv2.GaussianBlur(img, (ksize,ksize), sigma) 
    return lofil
    
def hipass(img, sigma, ksize):
    lofil = lopass(img, sigma, ksize)
    hifil = img-lofil
    """
    hifil -= hifil.min()
    cv2.imshow("win", hifil.astype(np.uint8))
    cv2.waitKey(0)
    """
    return hifil
    
def main():
    if len(sys.argv) != 3:
        print("usage: img_pyr_blend.py img1 img2")
        exit(0)

    # read images from input files        
    imgA = (cv2.imread(sys.argv[1])).astype(np.float32)
    imgB = (cv2.imread(sys.argv[2])).astype(np.float32)
    # convert to greyscale
    g_imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    g_imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    # set parameters:
    # kA & kB are the weights of each image when added
    kA = 1
    kB = 1
    # sigA and sigB are the stdev values for the Gaussian blur
    sigA = 5
    sigB = 3
    
    cv2.namedWindow("win")

    for i in range(1,24,2):
        print i
        lopassed = lopass(g_imgA, sigA, 9)
        hipassed = hipass(g_imgB, sigB, 9)
            
        hybrid = kA * lopassed + kB * hipassed
        
        if hybrid.min() < 0:
            hybrid -= hybrid.min()
                
        hybrid_norm = (hybrid *(255/hybrid.max())).astype(np.uint8)
    
        cv2.imshow("win", hybrid_norm)
        cv2.waitKey(0)

    cv2.imwrite("hybrid.jpg",hybrid_norm)

main()