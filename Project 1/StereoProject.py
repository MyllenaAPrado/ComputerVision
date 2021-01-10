#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:16:52 2020

@author: roberta e myllena
"""
import cv2
import numpy as np
import time
from pathlib import *
from optparse import OptionParser
import sys
import os.path
from matplotlib import pyplot as plt
#from scipy.linalg import inv,norm


op = OptionParser()

op.add_option("--r1",
              action="store_true", dest="r1_calibration",
              help="Requisito 1")
              
op.add_option("--r2",
              action="store_true", dest="r2_extrinsic",
              help="Requisito 2")
              
op.add_option("--r3",
              action="store_true", dest="r3_maps",
              help="Requisito 3")

op.add_option("--r4",
              action="store_true", dest="r4_tracker",
              help="Requisito 4")
              
def is_interactive():
	return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
	op.error("this script takes no arguments.")
	sys.exit(1)

print(__doc__)
op.print_help()
print()


def calibrateCamera(cam):
	"""
	Requisito 1: Calibração das câmeras
	:param cam: Nome do diretório com as imagens para calibração. 'Calibration1' ou 'Calibration2'
	:return cameraMatrix: Matriz de intrísecos
	"""
	cameraMatrix = []
	rvecs = []
	tvecs = []
	pointsWorld=[]
	pointsImg = []
	imgGray =[]
	filelist = []
	widthChess = 8
	heightChess = 6
	square_size = 0.293
    
	cv2.namedWindow(cam)
    
	#points in the real word
	points3D = np.zeros((heightChess*widthChess, 3), np.float32)
	points3D[:, :2] = np.mgrid[0:widthChess, 0:heightChess].T.reshape(-1, 2)
	points3D = points3D * square_size
    
	path_data = Path('./trabalho1_imagens/' + cam) 
    
	for f in path_data.iterdir():
		filelist.append(f)

	for file in filelist:
		imgPath = './' + str(file)
		img = cv2.imread(imgPath)

        #detect ChessboardCorners
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret,corners = cv2.findChessboardCorners(imgGray, (widthChess, heightChess), None)
		if(ret): #if found
				pointsWorld.append(points3D)
				#add ChessboardCorners points of image file
				corners2 = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
				#11,11
				pointsImg.append(corners2)
				img = cv2.drawChessboardCorners(img, (widthChess, heightChess), corners2, ret)

	cv2.imshow(cam, img)
	#Calibrate camera
	retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(pointsWorld, pointsImg, imgGray.shape[::-1], None, None)
	return cameraMatrix, distCoeffs

def addPosition(x, y):
	"""
	Função utilizada para o requisito 2. Guarda os valores das coordenadas do clique duplo na imagem.
	:param x, y: posições x e y do duplo clique na imagem
	"""
	pointsImage.append((x, y))

def getPosition(event,x,y,flags,param):
	"""
	Função utilizada para o requisito 2. Reconhecimento do evento de clique duplo na imagem.
	"""
	if event == cv2.EVENT_LBUTTONDBLCLK:
    		addPosition(x, y)

def captureFrame(cameraMatrix, distCoeffs, camera):
	"""
	Requisito 2: Estimativa da pose das câmeras
	:param cameraMatrix: Matriz de intrísecos
	:param distCoeffs: Coeficientes de distorção
	:param camera: Caminho do vídeo referente à câmera 1 ou 2
	:return R, tvec, C: Matriz de rotação, Vetor de translação, Coordenadas reais da câmera (Cx, Cy, Cz)
	"""
	cv2.namedWindow('videoCamera', cv2.WINDOW_NORMAL)

	global pointsImage
	#capture frame from video
	cap = cv2.VideoCapture(camera)

	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Error opening video stream or file")

	#get the frame 151
	cap.set(1, 150) 
	ret, frame = cap.read() 

	#get position of points in image
	cv2.imshow('videoCamera', frame)
	cv2.resizeWindow('videoCamera', 1200, 1200)
	
	cv2.setMouseCallback('videoCamera',getPosition)

	pointsWord = np.array([(0, 0, 0), (0, 1.4, 0), (2.6, 0, 0), (2.6, 1.4, 0)])
	pointsWord = np.float32(pointsWord[:, np.newaxis, :])
	
	pointsImage = []
	print('Selecione os 4 pontos na imagem com dois cliques, após a seleção digite "e"')

	while(1):    	
		k = cv2.waitKey(20) & 0xFF
		if k == ord('e'):
			break

	pointsImage = np.asarray(pointsImage, dtype=np.float32)
	retval, rvec, tvec = cv2.solvePnP(pointsWord, pointsImage, cameraMatrix, distCoeffs)
	
	#get matrix rotation 3X3
	rotation_mat = np.zeros(shape=(3, 3))
	R = cv2.Rodrigues(rvec, rotation_mat)[0]
    
	#estimating camera coordinates
	C = -np.matrix(R).T * np.matrix(tvec)
	return R, tvec, C

def pointsHomography(cam):
	"""
	Captura as coordenadas em pixel com clique duplo na imagem 
	:param cam: Caminho da imagem
	:return pointsImage: Coordenadas em pixels do clique duplo
	"""
	cv2.namedWindow('output', cv2.WINDOW_NORMAL)

	global pointsImage
	#pointsImage.clear()
	cv2.imshow("output", cam)       

	
	cv2.setMouseCallback('output',getPosition)
	
	pointsImage = []
	print('Selecione os 4 pontos na imagem com dois cliques, após a seleção digite "e"')

	while(1):    	
		k = cv2.waitKey(20) & 0xFF
		if k == ord('e'):
			break
    
	cv2.destroyAllWindows()

	pointsImage = np.asarray(pointsImage, dtype=np.float32)
	return pointsImage


### Essa função não foi utilizada na versão do código final - trocada pela calculateDisparity
def rectify(im1, im2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T):
    """
    Retifica as imagens im1 e im2 utilizandos o método stereoRectify da openCV
    :param im1, im2: Caminho da imagem 1, Caminho da imagem 2
    :param cameraMatrix1, cameraMatrix2: Matriz dos intrisecos da câmera 1, Matriz dos intrisecos da câmera 2
    :param distCoeffs1, distCoeffs2: Coeficientes de distorção da câmera 1, Coeficientes de distorção da câmera 2
    :param R: Matriz de rotação (saída do método stereoCalibrate da opencv) 
    :param T: Vetor de translação (saída do método stereoCalibrate da opencv)
    :return img: Imagem referente a camera 1 retificada 
    """
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (640, 480), R, T, alpha=0)
    
    mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (640, 480), cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (640, 480), cv2.CV_32F)
    img_rect1 = cv2.remap(im1, mapx1, mapy1, cv2.INTER_LINEAR)
    plt.imshow(img_rect1)
    plt.show()
    img_rect2 = cv2.remap(im2, mapx2, mapy2, cv2.INTER_LINEAR)
    plt.imshow(img_rect2)
    plt.show()
    
    # draw the images side by side
    total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1],3)
    img = np.zeros(total_size, dtype=np.uint8)
    img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
    img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
 
    # draw horizontal lines every 25 px accross the side by side image
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
 
    return img

def calculateDisparity(pointsImage_L, pointsImage_R, focal_l, imageLeft, imageRight):
    """
    Retifica as imagens imageLeft e imageRight utilizandos o método stereoRectifyUncalibrated da openCV
    :param pointsImage_L: Pontos em coordenadas em pixels da imagem da esquerda
    :param pointsImage_R: Pontos em coordenadas em pixels da imagem da direita
    :param focal_l: Distância focal da câmera da esquerda
    :param imageLeft: Caminho da imagem da esquerda
    :param imageRight: Caminho da imagem da esquerda
    :return disparity, depth, depthplot: Mapa de disparidade, mapa de profundidade, mapa de profundidade normalizado
    """
    
    imSL = cv2.resize(imageLeft, (640, 480))
    
    imSR = cv2.resize(imageRight, (640, 480))
    
    fundamental_matrix, inliers = cv2.findFundamentalMat(pointsImage_SLf, pointsImage_SRf, method=cv2.FM_LMEDS)
    
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pointsImage_SLf, pointsImage_SRf, fundamental_matrix, imgSize=(640, 480), threshold=0,)
    
    imgL_undistorted = cv2.warpPerspective(imSL, H1, (640, 480))
    #plt.imshow(imgL_undistorted,'gray')
    #plt.show()
    imgR_undistorted = cv2.warpPerspective(imSR, H2, (640, 480))
    #plt.imshow(imgR_undistorted,'gray')
    #plt.show()
    
    #  perspective transformation matrix
    #  https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
    Q = np.float64([[1,0,0,0],[0,-1,0,0],[0,0,focal_l*0.05,0], [0,0,0,1]])
	
    stereo = cv2.StereoBM_create(numDisparities=48, blockSize=5)
    disparity = stereo.compute(imgL_undistorted, imgR_undistorted)
    
    depth = cv2.reprojectImageTo3D(disparity, Q)
    depthplot = cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)	
    
    
    return disparity, depth, depthplot
    
    
def drawBox(img, rect):
	"""
 	Desenho do retângulo nos frames do video
	:param img: Imagem lida (frame do vídeo)
	:param rect: Coordenadas, largura e altura da caixa de seleção de objeto do tracker
	"""
    
	x, y, w, h = int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])
	cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    
def trackingObject(cam):
	"""
 	Função para executar o tracker do objeto. Seleciona no primeiro frame o objeto
     e aperta a tecla 'enter'
	:param cam: Caminho do vídeo
	:return trajectory: Array com os pontos em coordenadas de pixels do centro da caixa de seleção
	"""
	i=0
	cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
	cap = cv2.VideoCapture('./trabalho1_imagens/' + cam)

	ret, frame = cap.read()
	rect = cv2.selectROI('Tracking', frame, False)
	tracker.init(frame, rect)


	# Check if camera opened successfully
	if (cap.isOpened() == False):
		print("Error opening video stream or file")

	trajectory = []
	while(1):
		timer = cv2.getTickCount()
        # Descomentar as seguinte linhas para rodar o tracker para o vídeo 1
        #  : O vídeo 1 é 250fps é o vídeo 2 é 10fps por segundo. Necessário pois
        # o tracker no vídeo 1 pega muitos mais pontos do que o vídeo 2.
		# i = i + 25
		# cap.set(1, i) 
		ret, frame = cap.read()
		ret, rect = tracker.update(frame)
		center = (rect[0]+(rect[2]/2),rect[1]+(rect[3]/2))
		print(center)
		trajectory.append(center)
        
		fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
		
		if ret == True:
			drawBox(frame,rect)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.imshow('Tracking', frame)
		#k = cv2.waitKey(10) & 0xFF
			if cv2.waitKey(30) & 0xFF == ord('e'):
				break
		else:
			break
        
	cap.release()
	cv2.destroyAllWindows()
    
	return trajectory

def resize_video(cam, w, h):
	"""
 	Função para fazer o resize do vídeo cam no tamanho (w, h)
     Escreve o vídeo resultante no diretório './trabalho1_imagens/'
	:param cam: Caminho do vídeo
	:param w,h: resolução (w,h)    
	"""
	cap = cv2.VideoCapture('./trabalho1_imagens/' + cam + '.mp4')
    
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # XVID - não funciona no ubuntu
	out = cv2.VideoWriter('./trabalho1_imagens/' + cam + '_resized.mp4', fourcc, 250, (w, h))

	ret, frame = cap.read()
	if ret: # else it means the movie is over !
		res = cv2.resize(frame, (w, h))
		out.write(res)

if opts.r1_calibration:
	"""
 	Requisito 1
     Calibração dos intrínsecos das duas câmeras usadas para capturar as imagens 
     deste trabalho, usando as imagens disponíveis nos diretórios Calibration1 e 
     Calibration2, no diretório './trabalho1_imagens/'.
     Armazena os resultados em './resultados_r1/'.
	"""
	cameraMatrix1 = []   
	distCoeffs1 = []
	cameraMatrix1, distCoeffs1 = calibrateCamera('Calibration1')

	cameraMatrix2 = []
	distCoeffs2 = []
	cameraMatrix2, distCoeffs2 = calibrateCamera('Calibration2')
    
	if not os.path.exists('./resultados_r1'):
		os.makedirs('resultados_r1')
	
	np.save('./resultados_r1/params_intrisecos_cam1', cameraMatrix1)
	np.save('./resultados_r1/params_intrisecos_cam2', cameraMatrix2)
	np.save('./resultados_r1/coef_distorcao_cam1', distCoeffs1)
	np.save('./resultados_r1/coef_distorcao_cam2', distCoeffs2)
	
	print("Parâmetros intrísecos(K1):\n", cameraMatrix1, "\n")	
	print("Parâmetros intrísecos(K2):\n", cameraMatrix2, "\n")

	while(1):    	
		k = cv2.waitKey(20) & 0xFF
		if k == ord('e'):
			break

	cv2.destroyAllWindows()

if opts.r2_extrinsic:
	"""
 	Requisito 2
     Estimativa da pose das câmeras.
     Armazena os resultados em './resultados_r2/'.
	"""
	if os.path.isfile('./resultados_r1/params_intrisecos_cam1.npy') and os.path.isfile('./resultados_r1/params_intrisecos_cam2.npy') and os.path.isfile('./resultados_r1/coef_distorcao_cam1.npy') and os.path.isfile('./resultados_r1/coef_distorcao_cam2.npy'):
		cameraMatrix1 = np.load('./resultados_r1/params_intrisecos_cam1.npy')
		cameraMatrix2 = np.load('./resultados_r1/params_intrisecos_cam2.npy')
		distCoeffs1 = np.load('./resultados_r1/coef_distorcao_cam1.npy')
		distCoeffs2 = np.load('./resultados_r1/coef_distorcao_cam2.npy')
    		
		R_cam1 = []
		tvec_cam1 = []
		C_cam1 = []
		R_cam1, tvec_cam1, C_cam1 = captureFrame(cameraMatrix1,distCoeffs1,'./trabalho1_imagens/camera1.webm')
		print("Matriz de rotação - Camêra 1:\n", R_cam1, "\n")
		print("Vetor de translação - Camêra 1:\n", tvec_cam1, "\n")
		print("Coordenadas - Camêra 1:\n", C_cam1, "\n")
		
		R_cam2 = []
		tvec_cam2 = []
		C_cam2 = []
		R_cam2, tvec_cam2, C_cam2 = captureFrame(cameraMatrix1,distCoeffs2,'./trabalho1_imagens/camera2.webm')
		print("Matriz de rotação - Camêra 2:\n", R_cam2, "\n")
		print("Vetor de translação - Camêra 2:\n", tvec_cam2, "\n")
		print("Coordenadas - Camêra 2:\n", C_cam2, "\n")

		if not os.path.exists('./resultados_r2'):
			os.makedirs('resultados_r2')
		
		np.save('./resultados_r2/R_cam1', R_cam1)
		np.save('./resultados_r2/R_cam2', R_cam2)
		np.save('./resultados_r2/tvec_cam1', tvec_cam1)
		np.save('./resultados_r2/tvec_cam2', tvec_cam2)
		np.save('./resultados_r2/C_cam1', C_cam1)
		np.save('./resultados_r2/C_cam2', C_cam2)
		
	else:
		print ('Execute primeiramente o requisito 1 com a flag --r1')


if opts.r3_maps:
	"""
 	Requisito 3
     Mapa de disparidade e de profundidade.
     Plota os mapas e armazena os resultados em './resultados_r3/'.
	"""
	
    # Frames sincronizados utilizadas para a definição do mapa de disparidade
    # e de profundidade
	imgL = cv2.imread('./imagens_sinc/image1.png', 0)
	imgR = cv2.imread('./imagens_sinc/image2.png', 0)
    
    ################################
    # Descomentar para selecionar e utilizar outros pontos da imagem.
    # No mínimo 7 - método FM_LMEDS
    
    #pointsImage_R = pointsHomography(imSR)
	#pointsImage_Rf = np.float32(pointsImage_SR[:, np.newaxis, :])
	#np.save('./imagens_sinc/pointsR', pointsImage_SRf)
	#pointsImage_L = pointsHomography(imSL)
	#pointsImage_Lf = np.float32(pointsImage_SL[:, np.newaxis, :])
	#np.save('./imagens_sinc/pointsL', pointsImage_SLf)
    
    ################################

	if os.path.isfile('./resultados_r1/params_intrisecos_cam1.npy') and os.path.isfile('./resultados_r1/params_intrisecos_cam2.npy') and os.path.isfile('./resultados_r1/coef_distorcao_cam1.npy') and os.path.isfile('./resultados_r1/coef_distorcao_cam2.npy'):
		cameraMatrix2req1 = np.load('./resultados_r1/params_intrisecos_cam2.npy')
		focal_l_cam1 = cameraMatrix2req1[0][0]
    
		pointsImage_SRf = []
		pointsImage_SLf = []
		pointsImage_SRf = np.load('./imagens_sinc/pointsR.npy')
		pointsImage_SLf = np.load('./imagens_sinc/pointsL.npy')
    	# pontos: A, B, C, D, base do bloquinhos no carro, topo dos bloquinhos no carro, canto da janela do carrinho, meio do xadrez
    
		disparity, depth, depthplot = calculateDisparity(pointsImage_SLf, pointsImage_SRf, focal_l_cam1, imgL, imgR)
	
		if not os.path.exists('./resultados_r3'):
			os.makedirs('resultados_r3')
        
		cv2.imwrite('./resultados_r3/disparty_map.png', disparity)
		cv2.imwrite('./resultados_r3/depth_map.png', depth)
    
		plt.imshow(disparity, 'gray')
		plt.colorbar()
		plt.show()
        
		plt.imshow(depthplot, 'Blues')
		plt.colorbar()
		plt.show()
	else:
		print ('Execute primeiramente o requisito 1 com a flag --r1')
    
if opts.r4_tracker:
	"""
 	Requisito 4
     EStimativa das coordenadas reais da trajetória da base da pilha de blocos
     Plota as curvas referentes às coordenadas X, Y e Z.
	"""
    
    ################################
    # Bloco de código para resize dos vídeos e para obter informações
    # dos vídeos utilizados (fps e resolução)
    
	#resize_video('camera1_trim', 1280 , 720)
	#resize_video('camera2_trim', 1280 , 720)
    
	#cap = cv2.VideoCapture('./trabalho1_imagens/camera1_trim_resized.mp4')
	#fps = cap.get(cv2.CAP_PROP_FPS)
	#print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
	#ret, frame = cap.read()
	#if ret: # else it means the movie is over !	
	#	print('A')
	#	height, width = frame.shape[:2]
	#	print(width, ',', height)
    
    #################################
    # Bloco de código para rodar o tracker (https://www.youtube.com/watch?v=1FJWXOO1SRI)
    # O resultado foi salvo em './resultados_r4/', devido ao tempo
    # de processamento elevado para a câmera 1
	
	#tracker = cv2.TrackerMOSSE_create()
    
	#trajectory1 = trackingObject('camera1_trim.mp4')
    # para capturar a mesma quantidade de pontos da câmera 2,
    # descomentar as linhas indicadas na função
    
	#trajectory2 = trackingObject('camera2_trim.mp4')

	#print(trajectory1)
	#print(trajectory2)
    
	#np.save('./resultados_r4/trajectory1', trajectory1)
	#np.save('./resultados_r4/trajectory2', trajectory2)
    
    #################################
    
	trajectory1 = np.load('./resultados_r4/trajectory1.npy')
	#print(len(trajectory1))
	trajectory2 = np.load('./resultados_r4/trajectory2.npy')
	#print(len(trajectory2))
	C1 = np.load('./resultados_r2/C_cam1.npy')
	C2 = np.load('./resultados_r2/C_cam2.npy')

	cameraMatrix2req1 = np.load('./resultados_r1/params_intrisecos_cam2.npy')
	focal_l_cam1 = cameraMatrix2req1[0][0]
    
	baseline = cv2.norm(C1 - C2, cv2.NORM_L2)
	#print(baseline)
    
    #cam1 - right
    #cam2 - left
    
	cameraMatrix1 = np.load('./resultados_r1/params_intrisecos_cam1.npy')
	cameraMatrix2 = np.load('./resultados_r1/params_intrisecos_cam2.npy')	
	R_cam1 = np.load('./resultados_r2/R_cam1.npy')
	R_cam2 = np.load('./resultados_r2/R_cam2.npy')
	tvec_cam1 =	np.load('./resultados_r2/tvec_cam1.npy')
	tvec_cam2 =	np.load('./resultados_r2/tvec_cam2.npy')   
    
	trajectory1 = trajectory1[0:109]
    
	trajectory1_format = np.float32(trajectory1)
	trajectory2_format = np.float32(trajectory2)
    
	projection1 = np.matmul(cameraMatrix1, np.column_stack([R_cam1, tvec_cam1]))
	projection2 = np.matmul(cameraMatrix2, np.column_stack([R_cam2, tvec_cam2]))
    
	triangulated_points = []
	triangulated_points_real = []
	for i,point in enumerate(trajectory1_format):
		#print(trajectory2[i])
		triangulated_points = cv2.triangulatePoints(projection2, projection1, trajectory2[i], trajectory1[i])
		#print(triangulated_points)
		real_coordinates = []
		for j,item in enumerate(triangulated_points):
			if j < 3:
				#print(item)
				#print(triangulated_points[3])
				real_coordinates.append(item/triangulated_points[3])           
				#print(real_coordinates)
		triangulated_points_real.append(real_coordinates)

	#print(triangulated_points_real)
    
	X = []
	Y = []
	Z = []
    
	for point in triangulated_points_real:
		X.append(point[0])
		Y.append(point[1])
		Z.append(point[2])   
         
    ###################################
    # Método do professor vidal slide 14
	#for i,item in enumerate(trajectory2):
        # Xl = trajectory2[0] , Xr = trajectory1[0]
	#	Xsum = trajectory2[i][0] + trajectory1[i][0]
	#	Xdiff = abs(trajectory2[i][0] - trajectory1[i][0])
	#	X.append((baseline * Xsum)/(2 * Xdiff))        
		# Yl = trajectory2[1] , Yr = trajectory1[1]
	#	YSum = trajectory2[i][1] + trajectory1[i][1]
	#	Ydiff = abs(trajectory2[i][1] - trajectory1[i][1])
	#	Y.append((baseline * YSum)/(2 * Ydiff)) 
        # Z
	#	Z.append((baseline * focal_l_cam1)/(2* Xdiff))   
    ###################################

    # eixo x fps ou tempo
    # eixo y coordenada real
	
	x = np.arange(1, (12*10)-10)
	plt.title("Coordenada X da base da pilha de blocos")  
	plt.xlabel("frames")  
	plt.ylabel("Coordenada X")  
	plt.plot(x, X, color ="blue")  
	plt.show()
    
	x = np.arange(1, (12*10)-10)
	plt.title("Coordenada Y da base da pilha de blocos")  
	plt.xlabel("frames")  
	plt.ylabel("Coordenada Y")  
	plt.plot(x, Y, color ="blue")  
	plt.show()

	x = np.arange(1, (12*10)-10)
	plt.title("Coordenada Z da base da pilha de blocos")  
	plt.xlabel("frames")  
	plt.ylabel("Coordenada Z")  
	plt.plot(x, Z, color ="blue")  
	plt.show()