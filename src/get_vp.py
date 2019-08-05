"""
Methods to get vanishing points in an image. 

"""
import cv2
import os 
import numpy as np
import itertools as it 
from PIL import Image, ImageDraw
from pylsd.lsd import lsd
import matplotlib.pyplot as plt
import itertools


folderPath = "../data/Dataset/scripts/first_trial/right_folder"
files = [fn for fn in os.listdir(os.path.join(".",folderPath)) if fn.endswith('jpg')]



def lineThresh(line, thresh):
    if line is not None:
        pt1 = [int(line[0]), int(line[1])]
        pt2 = [int(line[2]), int(line[3])]
        # width = lines[i, 4]
        dx = int(line[2]) - int(line[0])
        dy = int(line[3]) - int(line[1])
        dx = dx
        dy = dy
        length = np.sqrt(dx*dx + dy*dy)
        if length >= thresh:
            return True
    return False



def drawLineSegments():
    global image,img
    rows,cols = image.shape
    lines = lsd(image)
    lines = removeFromBoxOfDoom(lines)
    lines = list(filter(lambda line: lineThresh(line,70),lines))
    # # lineFilter
    # for line in lines:
    #     cv2.line(image,tuple([int(line[0]),int(line[1])]), tuple([int(line[2]),int(line[3])]),(0,0,255),4)
    cv2.rectangle(img,(0,500),(1280,720),(0,0,255),4)
    return lines


def findBestCell(points,grid_size):
    global img,image
    image_height,image_width= image.shape
    # Grid dimensions
    grid_rows = (image_width // grid_size) + 1
    grid_columns = (image_height // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        cv2.rectangle(image, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 2)

        current_intersections = 0  # Number of intersections in the current cell
        for point in points:
            if cell_left < point[0] < cell_right and cell_bottom < point[1] < cell_top:
                current_intersections += 1
            # Current cell has more intersections that previous cell (better)
            if current_intersections > max_intersections:
                max_intersections = current_intersections
                best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)

    best_cell = tuple([int(best_cell[0]),int(best_cell[1])])
    edge1 = tuple([int(best_cell[0] -20),int(best_cell[1])])
    edge2 = tuple([int(best_cell[0] +20),int(best_cell[1])])
    cv2.circle(image,best_cell, 5,(170,200,255), -11)
    box = getBoxOfDoom()
    cv2.line(img,edge1, tuple(box[0]),(255,255,255), 4)
    cv2.line(img,edge2, tuple([box[1][0], box[0][1]]),(255,255,255), 4)



def getBoxOfDoom():
    box = [[0,400],[1280,720]]
    return box

def getValidBox():
    box = [[0,200],[1280,400]]
    return box

def isPointInsideBox(point):
    box = getValidBox()
    x1 = box[0][0]
    x2 = box[1][0]
    y1 = box[0][1]
    y2 = box[1][1]

    if point[0] >= x1 and point[0] <= x2 and point[1] >=y1 and point[1]<= y2:
        return True
    return False



def isLineOutsideBox(box, line):
    x1 = box[0][0]
    x2 = box[1][0]
    y1 = box[0][1]
    y2 = box[1][1]
    if line[0] >= x1 and line[2] <= x2:
        if line[1] >= y1 and line[3]<=y2:
            return False
    return True


def removeFromBoxOfDoom(lines):
    box = getBoxOfDoom()
    linesNew =[]
    linesNew = list(filter(lambda line:isLineOutsideBox(box,line),lines))
    return linesNew


def line(line):
    A = (line[1] - line[3])
    B = (line[2] - line[0])
    C = (line[0]*line[3] - line[2]*line[1])
    return A, B, -C


def lineIntersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def isPointInImage(point):
    global image
    rows,cols= image.shape
    if point[0] <0 or point[1] <0 or point[0]>rows or point[1]>cols or not isPointInsideBox(point):
        return False
    return True

# Find intersections between multiple lines (not line segments!)
def findIntersections(lines):
    intersections = []
    lines = list(map(line, lines))
    # lines = list(filter(lambda l : l[0]/l[1] <0, lines))
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            intersection = lineIntersection(line_1, line_2)
            if intersection:  # If lines cross, then add
                intersections.append(intersection)
    return intersections


def preProcess():
    global image 
    im_bw = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(im_bw)
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.bilateralFilter(image,9,50,50)
    # cv2.imshow("img", image)


def drawIntersections(points):
    global image
    for point in points:
        point = tuple([int(point[0]), int(point[1])])
        cv2.circle(img,tuple(point), 2,(0,0,255), -11)

# def filterLines(lines):
#     global image
#     for l in lines:
#         lp = line(l)
#         slope = -(lp[0]/lp[1])
#         if  slope <90:
#             cv2.line(image,tuple([int(l[0]),int(l[1])]), tuple([int(l[2]),int(l[3])]),(0,0,255),4)



def main():
    global image
    global img

    for i in range(1,100):
        file = os.path.join(folderPath,files[i])

        image = cv2.imread(file)
        img = image
        preProcess()

        lines = drawLineSegments()
        intersections = findIntersections(lines)
        intersections = list(filter(lambda point: isPointInImage(point),intersections))
        drawIntersections(intersections)

        findBestCell(intersections,100)

        cv2.circle(img,tuple([640,360]), 5,(255,255,255), -11)




        cv2.imshow("img", img)
        cv2.waitKey(delay = 0)


if __name__ == "__main__":
    main()