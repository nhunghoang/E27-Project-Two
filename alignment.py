import cv2
import numpy as np
import cvk2

def main():
	imgA = cv2.imread("Images/sunflower.jpg")
	imgB = cv2.imread("Images/bear34.jpeg")
	'''cv2.imshow("Image A", imgA)
	cv2.waitKey(0)	
	bwA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
	bwB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Image A BW", bwA)
	cv2.waitKey(0)'''	

	# grayscale based on petal color
	yellow = (0, 220, 248)
	diffs_per_channel = imgA.astype(np.float32) - yellow
	squared_dists = (diffs_per_channel**2).sum(axis=2)
	dists = np.sqrt(squared_dists)
	bwA = np.clip(dists, 0, 255).astype(np.uint8)
	
	#threshold, invert, morph, get petals
	ret, threshA = cv2.threshold(bwA, 125, 255, cv2.THRESH_BINARY)
	threshA = 255-threshA
	morphA = cv2.morphologyEx(threshA, cv2.MORPH_ERODE, np.ones((3,3),np.uint8))
	isolated = np.zeros_like(imgA)
	mask = morphA.view(np.bool)
	isolated[mask] = imgA[mask]
	cv2.imshow("Image A MORPH", morphA)
	cv2.waitKey(0)
	cv2.imshow("Image A MORPH", isolated)
	cv2.waitKey(0)

	#find the object as a contour and bound it by rectangle
	cpA = morphA.copy()
	contourImgA = np.zeros((cpA.shape[0], cpA.shape[1], 3), dtype='uint8')
	image, contours, hierarchy = cv2.findContours(cpA, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea)
	x,y,w,h = cv2.boundingRect(contours[-1])
	centered = np.zeros_like(imgA)
	height,width, ret = centered.shape
	rect = cv2.rectangle(contourImgA, (x,y), (x+w,y+h), (135,65,86))
	cv2.drawContours(contourImgA, contours, -1, (44,134,88), -1 )
	cv2.imshow("Image A CONT", contourImgA)
	cv2.waitKey(0)

	#shift object to center
	center_top = (height/2)-(h/2)
	center_bottom = center_top+h
	center_left = (width/2)-(w/2)
	center_right = center_left+w
	accum = 0
	for row in range(center_top, center_bottom+1):
		centered[row] = isolated[accum]
		accum += 1
	cv2.imshow("Shifted down",centered)
	cv2.waitKey(0)
	shift = x-center_left
	for col in range(center_left, center_right+1):
		for row in range(center_top, center_bottom+1):
			centered[row][col] = centered[row][col+shift]
	for col in range(center_right+1, width):
		for row in range(center_top,center_bottom+1):
			centered[row][col] = (0,0,0)
	centered[
	cv2.imshow("Shifted left",centered)
	cv2.waitKey(0)

	#next to do: change black background to (60, 137, 100)
main()
