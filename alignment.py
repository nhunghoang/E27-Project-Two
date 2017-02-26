import cv2
import numpy as np
import cvk2

def main():
	imgA = cv2.imread("Images/sunflower.jpg")
	imgB = cv2.imread("Images/zucker.jpeg")

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

	#shift flower to center
	center_top = (height/2)-(h/2)
	center_bottom = center_top+h
	center_left = (width/2)-(w/2)
	center_right = center_left+w
	accum = 0
	for row in range(center_top, center_bottom+1):
		centered[row] = isolated[accum]
		accum += 1
	shift = x-center_left
	for col in range(center_left, center_right+1):
		for row in range(center_top, center_bottom+1):
			centered[row][col] = centered[row][col+shift]
	for col in range(center_right+1, width):
		for row in range(center_top,center_bottom+1):
			centered[row][col] = (0,0,0)
	cv2.imwrite("centered_flower.jpg",centered)

	#face recognition
	#using tutorial: https://realpython.com/blog/python/face-recognition-with-python/
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	bwB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
    			bwB,
    			scaleFactor=1.1,
    			minNeighbors=5,
    			minSize=(15, 30),)
	for (x,y,w,h) in faces:
		cv2.rectangle(imgB, (x+50,y), (x+w,y+h), (0,0,0))
	cv2.imshow("detect face", imgB)
	cv2.waitKey(0)

	#generate face mask
	(x,y,w,h) = faces[0]
	mask = np.zeros_like(imgA)
	for row in range(y,y+h+1):
		for col in range(x+50,x+w+1):
			mask[row][col] = imgB[row][col]

	#shift face to center
	center_top = (height/2)-(h/2)
	center_bottom = center_top+h
	center_left = (width/2)-((w-50)/2)
	center_right = center_left+w
	shiftUD = y-center_top
	for row in range(center_top, center_bottom+1):
		mask[row] = mask[row+shiftUD]
	shiftLR = center_left-(x+50)
	maskcp = mask.copy()
	for col in range(center_left, center_right+1):
		for row in range(center_top, center_bottom+1):
			maskcp[row][col] = mask[row][col-shiftLR]
	for col in range(width):
		for row in range(center_bottom,height):
			maskcp[row][col] = (0,0,0)
	for col in range(0,center_left):
		for row in range(0,center_bottom):
			maskcp[row][col] = (0,0,0)
	cv2.imwrite("centered_face.jpg",maskcp)
main()
