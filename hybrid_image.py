"""
Program:  hybrid_image.py
Authors:  Daniel Green, Nhung Hoang, Richard Phillips
   Date:  2/26/2017
"""

import sys
import numpy as np
import cv2

def lopass(img, sigma, ksize):
    """
    Applies low-pass filter by convolving the image with a Gaussian kernel
    params:
        img - image to be blurred
        sigma - standard deviation of Gaussian kernel
        ksize - size of Gaussian kernel
    returns: input image with with low-pass filter applied
    """
    lofil = cv2.GaussianBlur(img, (ksize,ksize), sigma) 

    return lofil
    
def hipass(img, sigma, ksize):
    """
    Applies high-pass filter subtracting return of lopass() from the image
    params:
        img - unfiltered image
        sigma - standard deviation of Gaussian kernel (for lopass())
        ksize - size of Gaussian kernel (for lopass())
    returns: input image with with high-pass filter applied
    """
    lofil = lopass(img, sigma, ksize)
    hifil = img-lofil

    return hifil
    
def main():
    if len(sys.argv) != 3:
        print("usage: img_pyr_blend.py img1 img2")
        exit(0)

    # read images from input files; convert to float32 for operations
    imgA = (cv2.imread(sys.argv[1])).astype(np.float32)
    imgB = (cv2.imread(sys.argv[2])).astype(np.float32)
    # convert to greyscale
    g_imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    g_imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # set parameters:
    # kA & kB are the weights of each image when added
    kA = 1
    kB = 3
    # sigA and sigB are the stdev values for the Gaussian blur
    sigA = 5
    sigB = 3
    # ksize is size of Gaussian kernel for convolution
    ksize = 13
    
    # apply filter
    lopassed = lopass(g_imgA, sigA, ksize)
    hipassed = hipass(g_imgB, sigB, ksize)
    # sum lo-passed imgA and hipassed imgB to get hybrid        
    hybrid = kA * lopassed + kB * hipassed
    
    # ensure minimum value in hybrid is not less than 0
    if hybrid.min() < 0:
        hybrid -= hybrid.min()
    # make maximum value of hybrid 255 and scale all other values accordingly
    hybrid_norm = (hybrid *(255/hybrid.max())).astype(np.uint8)

    # display image
    cv2.namedWindow("win")
    cv2.imshow("win", hybrid_norm)
    cv2.waitKey(0)
    # save image
    cv2.imwrite("hybrid.jpg",hybrid_norm)

main()