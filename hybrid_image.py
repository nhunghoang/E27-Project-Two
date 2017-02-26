# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:06:19 2017

@author: Dan
"""
import sys
import numpy as np
import cv2

def lopass(img, sigma):
    
    return cv2.GaussianBlur(img, (7,7), sigma)
    
def hipass(img, sigma):
    return img - lopass(img, sigma)
    
def main():
    if len(sys.argv) != 3:
        print("usage: img_pyr_blend.py img1 img2")
        exit(0)
    
    imgA = cv2.imread(sys.argv[1])
    imgB = cv2.imread(sys.argv[2])
    
    kA = 3
    kB = 4
    
    sigA = 3
    sigB = 2

    lopassed = np.array(lopass(imgA, sigA)).astype(np.float32)
    hipassed = np.array(hipass(imgB, sigB)).astype(np.float32)

    hybrid = kA * lopassed + kB * hipassed

    hybrid_norm = (hybrid *(255/hybrid.max())).astype(np.uint8)

    cv2.namedWindow("win")
    cv2.imshow("win", hybrid_norm)
    cv2.waitKey(0)
    
    cv2.imwrite("hybrid.jpg",hybrid_norm)

main()