import cv2
import numpy as np

colors = [0,0,0]
#set how much the chroma key will change the image
threshold = 35

def ImageCamera(event,x,y,flags,param):
	global colors
	#checks mouse left button down condition
	if (event == cv2.EVENT_LBUTTONDOWN):
		colors = frame[y,x]
		
#change the pixels similary with the select with mouse
def changeImage():
	global colors, threshold
	
	#read the image to background
	background = cv2.imread('/home/myllena/Documents/Pvc/ComputerVision/Chroma-key/im.jpg')
	
	#set interval of color for change in the image captured
	lower = colors - threshold
	upper = colors + threshold

	image_copy = np.copy(frame)

	#create a mask 
	mask = cv2.inRange(image_copy, lower, upper)
	masked_image = np.copy(image_copy)
	masked_image[mask != 0] = [0, 0, 0]

	#cut the image in the bakcgroung
	crop_background = background[0:480, 0:640]
	crop_background[mask == 0] = [0, 0, 0]

	#final image with the chromaKey 
	final_image = crop_background + masked_image
	cv2.imshow('ImageCamera',final_image)


#configure the window and read the mouse
cv2.namedWindow('ImageCamera')
cv2.setMouseCallback('ImageCamera',ImageCamera)

capture = cv2.VideoCapture(0)

while(True):

    ret, frame = capture.read()
    #select for show the normal image or the chroma key image
    if(colors[0]):
    	changeImage()
    else:
    	cv2.imshow('ImageCamera', frame)

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()