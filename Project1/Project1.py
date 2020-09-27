import cv2
import numpy as np
import time

cv2.namedWindow('ImageCamera1')
cv2.namedWindow('ImageCamera2')

def CalibrateCamera1():
	global cameraMatrix1, rvecs1, tvecs1;
	pointsWorld=[]
	pointsImg = []
	imgGray =[]
	widthChess = 8
	heightChess = 6
	square_size = 0.20

	#points in the real word
	points3D = np.zeros((heightChess*widthChess, 3), np.float32)
	points3D[:, :2] = np.mgrid[0:widthChess, 0:heightChess].T.reshape(-1, 2)
	points3D = points3D * square_size

	#read the imagens
	for i in range(31):
		#read the imagens from file
		imgPath = '/home/myllena/Documents/Pvc/ComputerVision/Project1/Calibration1/Image' + str(i+1) + '.jpg'
		img = cv2.imread(imgPath)

		#detect ChessboardCorners
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret,corners = cv2.findChessboardCorners(imgGray, (widthChess, heightChess), None)
		if(ret): #if found
			pointsWorld.append(points3D)
			#add ChessboardCorners points of image file
			corners2 = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
			pointsImg.append(corners2)
			img = cv2.drawChessboardCorners(img, (widthChess, heightChess), corners2, ret)
			
	cv2.imshow('ImageCamera1', img)
	#Calibrate camera
	retval, cameraMatrix1, distCoeffs, rvecs1, tvecs1 = cv2.calibrateCamera(pointsWorld, pointsImg, imgGray.shape[::-1], None, None)

def CalibrateCamera2():
	global cameraMatrix2, rvecs2, tvecs2;
	pointsWorld=[]
	pointsImg = []
	imgGray =[]
	widthChess = 8
	heightChess = 6
	square_size = 0.20

	#points in the real word
	points3D = np.zeros((heightChess*widthChess, 3), np.float32)
	points3D[:, :2] = np.mgrid[0:widthChess, 0:heightChess].T.reshape(-1, 2)
	points3D = points3D * square_size

	#read the imagens
	for i in range(15):
		#read the imagens from file
		imgPath = '/home/myllena/Documents/Pvc/ComputerVision/Project1/Calibration2/Image' + str(i+1) + '.jpg'
		img = cv2.imread(imgPath)

		#detect ChessboardCorners
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret,corners = cv2.findChessboardCorners(imgGray, (widthChess, heightChess), None)
		if(ret): #if found
			pointsWorld.append(points3D)
			#add ChessboardCorners points of image file
			corners2 = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
			pointsImg.append(corners2)
			img = cv2.drawChessboardCorners(img, (widthChess, heightChess), corners2, ret)

	cv2.imshow('ImageCamera2', img)
	#Calibrate camera
	retval, cameraMatrix2, distCoeffs, rvecs2, tvecs2 = cv2.calibrateCamera(pointsWorld, pointsImg, imgGray.shape[::-1], None, None)

#variable calibration
cameraMatrix1 = []
cameraMatrix2 = []
rvecs1 = []
rvecs2 = []
tvecs1 = []
tvecs2 = []

CalibrateCamera1()
CalibrateCamera2()

print("Parâmetros intrísecos(K1):\n", cameraMatrix1, "\n")
print("Parâmetros intrísecos(K2):\n", cameraMatrix2, "\n")

while(True):
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()