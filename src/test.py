import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np

folderPath = "Dataset/scripts/last_trial/right_folder"
files = [fn for fn in os.listdir(os.path.join(".",folderPath)) if fn.endswith('jpg')]
f, axarr = plt.subplots(2,2)




def perspective(img, src, dst):

	rows,cols,ch = img.shape
	cropped = img[int(rows/2):int(rows),:]
	h,s,v = cv2.split(cropped)
	edgemap = cv2.Canny(s,200,100)
	rows,cols = s.shape
	src_pts = np.array([[10,120], [600,10],[900,120 ],[1200,120]], dtype = "float32")
	dst_pts = np.array([[10,cols],[10,0],[1000,0],[1000,cols]], dtype = "float32")
	H = cv2.getPerspectiveTransform(src_pts,dst_pts)
	dst = cv2.warpPerspective(cropped, H,(int(rows),int(cols) ) )
	return edgemap,dst


def lineswithHough(edgemap,img):
	lines = cv2.HoughLines(edegemap,1,np.pi/180,200)
	for rho,theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

def ProbHough(edgemap,img):
	minLineLength = 100
	maxLineGap = 100
	lines = cv2.HoughLinesP(edegemap,1,np.pi/180,100,minLineLength,maxLineGap)
	for x1,y1,x2,y2 in lines[0]:
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)





color = []
for i in range(1,50):
	img = cv2.imread(os.path.join(folderPath,files[i]))





	print(img.shape)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	rows,cols,ch =img.shape

	# ch,cs,cv = np.mean(h),np.mean(s),np.mean(v)
	# lower = np.array([ch-4,cs-4,cv-4])
	# upper = np.array([ch+4, cs+4,cv+4])
	# mask = cv2.inRange(hsv,lower,upper)

	# im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# cropped = img[int(0):int(rows/2),:]
	# # cv2.imshow('cropped',cropped)
	# h,s,v = cv2.split(cropped)
	# cs = np.mean(cropped)
	# ch,cs,cv = np.mean(h),np.mean(s),np.mean(v)
	# ret, thresh = cv2.threshold(h, int(ch)-40, int(ch)+40, 0)
	# contours, heir = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5] # get largest five contour area
	# cv2.drawContours(img, cnts, -1, (0,255,0), 0)
	
	# mask = np.zeros(img.shape, np.uint8)
	# cv2.drawContours(mask, cnts, -1, (255), 0)

	# mask = cv2.bitwise_not(mask)
	# cv2.imshow('mask',mask)
	# im_new = cv2.bitwise_and(mask,img)

	# res = cv2.addWeighted(img,0.1,im_new,0.4,0)








	# p,r,n = contours.shape
	# contours = contours.reshape(r,n)
	# print(contours.shape)

	# rects = []
	# for c in contours:
	# 	print(c.shape)
	# 	peri = cv2.arcLength(c, True)
	# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# 	x, y, w, h = cv2.boundingRect(approx)
	# 	if h >= 15:
	# 	# if height is enough
	# 	# create rectangle for bounding

	# 		rect = (x, y, w, h)
	# 		rects.append(rect)
	# 		cv2.rectangle(roi_copy, (x, y), (x+w, y+h), (0, 255, 0), 1);

	# mask2 = cv2.bitwise_not(mask1)
	# cv2.imshow("mask2",mask2)
	# res1 = cv2.bitwise_and(hsv,hsv,mask=mask2)
	# cv2.imshow('result',res1)

	# axarr[0,0].imshow(img)
	# axarr[0,1].imshow(h)
	# axarr[1,0].imshow(s)
	# axarr[1,1].imshow(v)
	# plt.pause(10)
	# th =  cv2.adaptiveThreshold(s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#        cv2.THRESH_BINARY,3,5)
	# thresh = 250
	# s = s[thresh:rows,:]
	# cv2.imshow('cropped',s)
	# rows,cols =s.shape
	src_pts = np.array([[10,120], [600,10],[900,120 ],[1200,120]], dtype = "float32")
	dst_pts = np.array([[10,cols],[10,0],[1000,0],[1000,cols]], dtype = "float32")

	edegemap, dst = perspective(img,src_pts,dst_pts)

	# color_fr = 




	cv2.imshow('edegemap',edegemap)
	# cv2.imshow('result',res)
	cv2.imshow('lines',dst)
	cv2.waitKey(0)
# plt.show()



plt.close('all')
