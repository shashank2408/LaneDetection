import cv2
import os 
import numpy as np

folderPath = "../data/Dataset/scripts/last_trial/right_folder"
files = [fn for fn in os.listdir(os.path.join(".",folderPath)) if fn.endswith('jpg')]


def getMean(channel):
	rows,cols =channel.shape
	cropped = channel[int(rows-rows/4):int(rows),:]
	cs = np.mean(cropped)
	return cs



def getmask(channel):
	rows,cols =channel.shape
	ch = getMean(channel)
	ret, thresh = cv2.threshold(channel, int(ch)-10, int(ch)+10, 0)
	contours, heir = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:1] # get largest five contour area
	mask = np.zeros(channel.shape, np.uint8)
	cv2.drawContours(channel, cnts, -1, (255), 0)
	# cnts = np.array(cnts)
	# print(cnts[0].tolist())
	return channel




def drawContours():
	global image

	h,s,v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))


	# cs = np.mean(cropped)
	# ch,cs,cv = np.mean(h),np.mean(s),np.mean(v)
	layer1 = getmask(h)
	layer2 = getmask(s)
	layer3 = getmask(v)
	cv2.imshow('layer1',layer1)
	cv2.imshow('layer2',layer2)
	cv2.imshow('layer3',layer3)
	# layer = layer1 + layer2 + layer3
	# # cv2.imshow('mask',mask)
	# cv2.imshow('combined', layer)
	# im_new = cv2.bitwise_and(mask,image)
	# res = cv2.addWeighted(image,0.6,im_new,1,0)
	# cv2.imshow("res", res)



def adjustMinT(v):
	global minT
	minT = v
	cannyEdge()


def adjustMaxT(v):
	global maxT
	maxT = v
	cannyEdge()


def cannyEdge():
	global image, minT, maxT
	edge = cv2.Canny(image = image, 
	    threshold1 = minT, 
	    threshold2 = maxT)

	cv2.imshow(winname = "edges", mat = edge)


def removeShadows():
	global image

	rgb_planes = cv2.split(image)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
		dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
		bg_img = cv2.medianBlur(dilated_img, 19)
		diff_img = 255 - cv2.absdiff(plane, bg_img)
		norm_img = np.zeros(diff_img.shape, np.uint8)
		cv2.normalize(diff_img, norm_img, alpha=4, beta=75, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		result_planes.append(diff_img)
		result_norm_planes.append(norm_img)

	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)

	return result, result_norm




def createWindowForEdge():
	minT = 30
	maxT = 150
	cv2.namedWindow(winname = "edges", flags = cv2.WINDOW_NORMAL)
	cv2.createTrackbar("minT", "edges", minT, 255, adjustMinT)
	cv2.createTrackbar("maxT", "edges", maxT, 255, adjustMaxT)
	cannyEdge()

for i in range(1,20):
	image = cv2.imread(os.path.join(folderPath,files[i]))
	# image = cv2.GaussianBlur(image,(5,5),0)
	# image = image
	res, res_norm = removeShadows()
	# cv2.imshow('res',res)
	# cv2.imshow('res_norm',res_norm)

	image = cv2.addWeighted(image,0.6,res,0.4,0)
	# cv2.imshow('fin',fin)
	drawContours()
	cv2.waitKey(delay = 0)