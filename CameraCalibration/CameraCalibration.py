import cv2
import numpy as np
import time

cv2.namedWindow('ImageCamera')

widthChess = 9
heightChess = 6
square_size = 0.21
pointsWorld=[]
pointsImg = []
imgGray =[]

def getPoints(widthChess, heightChess, square_size): 
	global pointsWorld, pointsImg, imgGray
	#points in the real word
	points3D = np.zeros((heightChess*widthChess, 3), np.float32)
	points3D[:, :2] = np.mgrid[0:widthChess, 0:heightChess].T.reshape(-1, 2)
	points3D = points3D * square_size

	imgPath = '/home/myllena/Documents/Pvc/ComputerVision/CameraCalibration/image1.jpg'
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
		cv2.imshow('ImageCamera', img)


#get poitns of the world and the image
getPoints(widthChess, heightChess, square_size)

#Calibrate Camera
retval, K, distCoeffs, rvecs, T = cv2.calibrateCamera(pointsWorld, pointsImg, imgGray.shape[::-1], None, None)

#get matrix rotation 3X3
rotation_mat = np.zeros(shape=(3, 3))
R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]

#get matrix projection
projection = np.matmul(K, np.column_stack([R, T[0]]))

print("Parâmetros intrísecos(K):\n", K, "\n")
print("Parâmetros de distorção:\n",distCoeffs, "\n")
print("Vetor de translação (T): \n",T, "\n")
print("Matriz de rotação(R):\n", R, "\n")
print("Matriz de projeção(P):\n ", projection)


cv2.destroyAllWindows()