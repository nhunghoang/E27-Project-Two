import cv2
import numpy as np

def pyr_build(img, N=6):
    """takes an 8-bit per channel RGB or grayscale image as input, and
    which outputs a list lp of Laplacian images (stored in numpy.float32 format)"""
    img = np.array(img, dtype=np.float32)
    orig_shape = img.shape
    # List to add pyramids on to
    lp = []

    G_i = img.copy()
    print G_i.shape
    for i in range(N):
        if i == N-1:
            # If we're at the smallest size pyramid image
            break
            # cv2.imshow('cvland', 0.5 + 0.5 *(L / np.abs(L).max()))
            # cv2.waitKey(0)
        else:
            G_i_plus = cv2.pyrDown(G_i)
            print G_i_plus.shape
            G_i_up = np.zeros(img.shape)
            G_i_up = cv2.pyrUp(G_i_plus, G_i_up)
            L = G_i - G_i_up
            lp.append(L)
            G_i = G_i_plus.copy()
            # cv2.imshow( 'cvland', 0.5 + 0.5 *(L / np.abs(L).max()))
            # cv2.waitKey(0)
            # print 'shape G_i_plus', G_i_plus.shape
    lp.append(G_i)
    L = lp[-1]
    return lp

def pyr_reconstruct(lp):
    Rn = lp[-1]
    for i in reversed(lp[0:-1]):
        R_i_up = np.zeros(i.shape)
        R_i_up = cv2.pyrUp(Rn, R_i_up)
        R_i_minus = R_i_up + i
        Rn = R_i_minus.copy() 
    #cv2.imshow('cvland', np.array(R_i_minus, dtype=np.uint8))
    #cv2.waitKey(0)
    return R_i_minus

def main():


    image1 = "Images/penguin.png"
    base_image1 = cv2.imread(image1)
    lp1 = pyr_build(base_image1)
    reconstructed1 = pyr_reconstruct(lp1)

    image2 = "Images/bearface.jpg"
    base_image2 = cv2.imread(image2)
    lp2 = pyr_build(base_image2)
    reconstructed2 = pyr_reconstruct(lp2)

if __name__=='__main__':
    main()
