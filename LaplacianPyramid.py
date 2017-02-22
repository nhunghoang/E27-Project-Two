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
            lp.append(G_i)
            L = lp[-1]
            cv2.imshow('cvland', 0.5 + 0.5 *(L / np.abs(L).max()))
            cv2.waitKey(0)
        else:
            G_i_plus = cv2.pyrDown(G_i)
            print G_i_plus.shape
            G_i_up = np.zeros(img.shape)
            G_i_up = cv2.pyrUp(G_i_plus, G_i_up)
            L = G_i - G_i_up
            lp.append(L)
            G_i = G_i_plus.copy()
            cv2.imshow( 'cvland', 0.5 + 0.5 *(L / np.abs(L).max()))
            cv2.waitKey(0)
            print 'shape G_i_plus', G_i_plus.shape

        """cv2.error: /Users/travis/miniconda3/conda-bld/work/opencv-3.1.0/modules'
        /imgproc/src/pyramids.cpp:994: error: (-215)
        std::abs(dsize.width - ssize.width*2) == dsize.width % 2 &&
        std::abs(dsize.height - ssize.height*2) == dsize.height % 2 in function pyrUp_
        """


    print lp

        # cv2.imshow( 'cvland', cv2.pyrUp(np.array(G_i_plus, dtype=np.uint8)))
        # cv2.waitKey(0)
        # print 'shape G_i_plus', G_i_plus.shape
        #

def main():


    image = "Images/penguin.png"
    base_image = cv2.imread(image)
    pyr_build(base_image)


if __name__=='__main__':
    main()