import cv2 as cv
from filePicker import OpenFile

def showImg(filename):
	img = cv.imread(filename,cv.IMREAD_COLOR)
	img_gray = cv.resize((255-img),(700,700))
	img = cv.resize(img,(700,700))
	img_gray = cv.cvtColor(img_gray,cv.COLOR_BGR2GRAY)

	_,thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

	_, cnts,_ = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

	rects = [cv.boundingRect(contour) for contour in cnts]

	subimages = []

	for (x,y,w,h) in rects:
		cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
		subimages.append(img[y: y+h, x: x+w])


	cv.imshow("Original",img)
	cv.waitKey(0)
	cv.destroyAllWindows()


filename = OpenFile()
showImg(filename)