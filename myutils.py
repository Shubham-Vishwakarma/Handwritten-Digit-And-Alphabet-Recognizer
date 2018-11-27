import cv2 as cv
import math
from scipy import ndimage
import numpy as np

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv.warpAffine(img,M,(cols,rows))
    return shifted


def pre_process_image(filename=None, img=None):
	if filename is not None:
		img = cv.imread(filename,cv.IMREAD_COLOR)
	img_gray = cv.resize((255-img),(28,28))
	img_gray = cv.cvtColor(img_gray,cv.COLOR_BGR2GRAY)
	_, img_gray = cv.threshold(img_gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

	while np.sum(img_gray[0]) == 0:
	    img_gray = img_gray[1:]

	while np.sum(img_gray[:,0]) == 0:
	    img_gray = np.delete(img_gray,0,1)

	while np.sum(img_gray[-1]) == 0:
	    img_gray = img_gray[:-1]

	while np.sum(img_gray[:,-1]) == 0:
	    img_gray = np.delete(img_gray,-1,1)

	rows,cols = img_gray.shape

	if rows > cols:
	    rows = 20
	    factor = rows/cols
	    cols = int(round(cols*factor))
	    img_gray = cv.resize(img_gray, (cols,rows))
	else:
	    cols = 20
	    factor = cols/rows
	    rows = int(round(rows*factor))
	    img_gray = cv.resize(img_gray, (cols, rows))

	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	img_gray = np.lib.pad(img_gray,(rowsPadding,colsPadding),'constant')

	shiftx,shifty = getBestShift(img_gray)
	shifted = shift(img_gray,shiftx,shifty)
	img_gray = shifted

	return img_gray

def get_subimages(filename):
	img = cv.imread(filename,cv.IMREAD_COLOR)
	img_gray = cv.resize((255-img),(700,700))
	img = cv.resize(img,(700,700))
	img_gray = cv.cvtColor(img_gray,cv.COLOR_BGR2GRAY)

	_,thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

	_, cnts,_ = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	rects = [cv.boundingRect(contour) for contour in cnts]

	subimages = []

	for (x,y,w,h) in rects:
		subimages.append(img[y:y+h, x:x+w])
		
	cv.destroyAllWindows()

	return subimages, rects


def show_prediction(filename,predicted):
	img = cv.imread(filename,cv.IMREAD_COLOR)
	img_gray = cv.resize((255-img),(280,280))
	img = cv.resize(img,(280,280))
	img_gray = cv.cvtColor(img_gray,cv.COLOR_BGR2GRAY)

	_,thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

	_, cnts,_ = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	rects = [cv.boundingRect(contour) for contour in cnts]

	for (x,y,w,h) in rects:
		cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
		cv.putText(img,str(predicted),(x,y),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

	cv.imshow("Image",img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def show_multiple_prediction(filename, results):
	img = cv.imread(filename,cv.IMREAD_COLOR)
	img_gray = cv.resize((255-img),(700,700))
	img = cv.resize(img,(700,700))
	img_gray = cv.cvtColor(img_gray,cv.COLOR_BGR2GRAY)

	_,thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

	_, cnts,_ = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	rects = [cv.boundingRect(contour) for contour in cnts]

	for (x,y,w,h), predicted in zip(rects,results):
		cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
		cv.putText(img,str(predicted),(x,y),cv.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

	cv.imshow("Image",img)
	cv.waitKey(0)
	cv.destroyAllWindows()