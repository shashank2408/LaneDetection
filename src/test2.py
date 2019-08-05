import cv2
import os 
import numpy as np
import itertools as it 
from PIL import Image, ImageDraw
from pylsd.lsd import lsd
import matplotlib.pyplot as plt


folderPath = "../data/Dataset/scripts/last_trial/right_folder"
files = [fn for fn in os.listdir(os.path.join(".",folderPath)) if fn.endswith('jpg')]


#http://ppniv14.irccyn.ec-nantes.fr/material/session2/Seo/presentation.pdf

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
    layer = layer1 + layer2 + layer3
    # cv2.imshow('mask',mask)
    cv2.imshow('combined', layer)
    im_new = cv2.bitwise_and(mask,image)
    res = cv2.addWeighted(image,0.6,im_new,1,0)
    cv2.imshow("res", res)



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
    print(edge)
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


# Perform edge detection
def hough_transform(img):
    global image 
    # img = image
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    kernel = np.ones((11, 11), np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    edges = cv2.Canny(opening, 100, 200, apertureSize=3)  # Canny edge detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # Hough line detection

    hough_lines = []
    # Lines are represented by rho, theta; converted to endpoint notation
    # if lines is not None:
    #   for line in lines:
    #       hough_lines.extend(list(it.starmap(endpoints, line)))
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.imshow("img", image)




def endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))

    return ((x_1, y_1), (x_2, y_2))



def createWindowForEdge():
    global minT,maxT 
    minT= 30
    maxT = 150
    cv2.namedWindow(winname = "edges", flags = cv2.WINDOW_NORMAL)
    cv2.createTrackbar("minT", "edges", minT, 255, adjustMinT)
    cv2.createTrackbar("maxT", "edges", maxT, 255, adjustMaxT)



def lineSegementDetect():
    global image
    lsd = cv2.createLineSegmentDetector(0)
    lines = cv2.detect(image)
    cv2.drawSegments(image,lines)
    cv2.imshow("lines",image)

def extrapolateLines():
    global image 
    rows,cols,ch = image.shape
    VP = tuple([int(cols/2),int(rows/2)])
    endPoints = [[0,400], [1200,400]]
    cv2.line(image,VP,tuple(endPoints[0]),(0,0,255),4)
    cv2.line(image,VP,tuple(endPoints[1]),(0,0,255),4)

def preProcess():
    global image 
    im_bw = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(im_bw)
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.bilateralFilter(image,9,50,50)
    # cv2.imshow("img", image)
# def computeDerivatives 


def drawLineSegments(file):
    global image
    rows,cols = image.shape
    # gray = np.asarray(image.convert('L'))
    # gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    lines = lsd(image)
    # draw = ImageDraw.Draw(image)
    if lines is not None:
        for line in lines:
            # print(line)
            pt1 = [int(line[0]), int(line[1])]
            pt2 = [int(line[2]), int(line[3])]
            # width = lines[i, 4]
            dx = int(line[2]) - int(line[0])
            dy = int(line[3]) - int(line[1])
            dx = dx
            dy = dy
            length = np.sqrt(dx*dx + dy*dy)
            if length > 50:
                print(length)
                cv2.line(image,tuple(pt1), tuple(pt2),(0,0,255),4)

    cv2.imshow("img", image)



def main():
    global image

    # createWindowForEdge()
    for i in range(1,20):
        file = os.path.join(folderPath,files[i])
        image = cv2.imread(file)
        # image = cv2.GaussianBlur(image,(5,5),0)
        # res, res_norm = removeShadows()
        # cv2.imshow('res',res)
        # cv2.imshow('res_norm',res_norm)
        preProcess()
        # Sobel()
        # cannyEdge()
        # image = cv2.addWeighted(image,0.6,res,0.4,0)
        # cv2.imshow('fin',fin)
        # drawContours()
        # extrapolateLines()
        # lineSegementDetect()
        # hough_transform(image)

        drawLineSegments(file)
        cv2.waitKey(delay = 0)


if __name__ == "__main__":
    main()