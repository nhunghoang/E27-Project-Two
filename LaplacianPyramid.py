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

def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)
    return A + alpha * (B-A)

def laplacian_blend(A, B, alpha):
    # Convert to prefered format for safer math on images
    A, B = np.array(A,dtype=np.float32), np.array(B, dtype=np.float32)
    # Get both pyramids
    A_pr, B_pr = pyr_build(A), pyr_build(B)
    layer_blend = []
    # Use mask on both pyramids (continuosly resizing, of course)
    for a, b in zip(A_pr, B_pr):
        assert( a.shape[:2] == b.shape[:2] )
        height, width = a.shape[:2]
        alpha = cv2.resize(alpha, (width, height), interpolation = cv2.INTER_AREA)
        layer_blend.append(alpha_blend(a,b, alpha))

    # Ta-da
    return np.clip(pyr_reconstruct(layer_blend),0,255)


def main():

    image1 = "centered_flower.jpg"
    base_image1 = cv2.imread(image1)
    lp1 = pyr_build(base_image1)
    reconstructed1 = pyr_reconstruct(lp1)

    image2 = "centered_face.jpg"
    base_image2 = cv2.imread(image2)
    lp2 = pyr_build(base_image2)
    reconstructed2 = pyr_reconstruct(lp2)

    width, height, _ = base_image2.shape
    mask = np.zeros((width, height), dtype=np.uint8)
    cv2.ellipse(mask, (height/2, width/2), (140, 100), 90, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    mask_blurred = cv2.GaussianBlur(mask, (0,0), 5)
    alpha = mask_blurred.astype(np.float32) / 255.0

    cv2.imshow('blended', np.array(alpha_blend(base_image1, base_image2, alpha), dtype=np.uint8))
    cv2.waitKey(0)

    cv2.imshow('blended', np.array(laplacian_blend(base_image1, base_image2, alpha), dtype=np.uint8))
    cv2.waitKey(0)

if __name__=='__main__':
    main()
