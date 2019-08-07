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


folderPath = "../data/Dataset/scripts/last_trial/right_folder"
files = [fn for fn in os.listdir(os.path.join(".",folderPath)) if fn.endswith('jpg')]

class GetVanishingPoint:

    def __init__(self):
        self.thresholdLine = 70

    def setImage(self,image):
        self.origImage = image
        self.image = self.preProcess() 
        [self.cols,self.rows] = self.image.shape


    def preProcess(self):
        im_bw = cv2.cvtColor(self.origImage,cv2.COLOR_RGB2GRAY)
        image = cv2.equalizeHist(im_bw)
        image = cv2.GaussianBlur(image,(3,3),0)
        image = cv2.bilateralFilter(image,9,50,50)
        return image

    def lineThresh(self,line, thresh):
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


    def drawLineSegments(self,drawLines=False):
        lines = lsd(self.image)
        lines = self.removeFromBoxOfDoom(lines)
        lines = list(filter(lambda line: self.lineThresh(line,self.thresholdLine),lines))
        # # lineFilter
        if drawLines:
            for line in lines:
                cv2.line(self.origImage,tuple([int(line[0]),int(line[1])]), \
                    tuple([int(line[2]),int(line[3])]),(0,0,255),4)
        return lines


    def findBestCell(self,points,grid_size,vis_grid=False):

        # Grid dimensions
        grid_rows = (self.cols // grid_size) + 1
        grid_columns = (self.rows // grid_size) + 1

        # Current cell with most intersection points
        max_intersections = 0
        best_cell = (0.0, 0.0)

        for i, j in itertools.product(range(grid_rows), range(grid_columns)):
            cell_left = i * grid_size
            cell_right = (i + 1) * grid_size
            cell_bottom = j * grid_size
            cell_top = (j + 1) * grid_size
            if vis_grid:
                cv2.rectangle(image, (cell_left, cell_bottom), (cell_right, cell_top)\
                    , (0, 0, 255), 2)
            current_intersections = 0  # Number of intersections in the current cell
            for point in points:
                if cell_left < point[0] < cell_right and cell_bottom < point[1] < cell_top:
                    current_intersections += 1
                # Current cell has more intersections that previous cell (better)
                if current_intersections > max_intersections:
                    max_intersections = current_intersections
                    best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
        return best_cell
       
    def visualizePath(self,best_cell):
        best_cell = tuple([int(best_cell[0]),int(best_cell[1])])
        edge1 = tuple([int(best_cell[0] -20),int(best_cell[1])])
        edge2 = tuple([int(best_cell[0] +20),int(best_cell[1])])
        cv2.circle(self.origImage,best_cell, 5,(170,200,255), -11)
        box = self.getBoxOfDoom()
        cv2.line(self.origImage,edge1, tuple(box[0]),(255,255,255), 4)
        cv2.line(self.origImage,edge2, tuple([box[1][0], box[0][1]]),(255,255,255), 4)

    def getBoxOfDoom(self,visulaise=False):
        box = [[0,int(self.cols-self.cols/3)],[self.rows,self.cols]]
        if visulaise:
            self.drawBox(box)
        return box

    def getValidBox(self,visulaise=False):
        box = [[0,int(self.cols/3)],[self.rows,int(self.cols-self.cols/3)]]
        if visulaise:
            self.drawBox(box)
        return box

    def isPointValid(self,point):
        box = self.getValidBox()
        x1 = box[0][0]
        x2 = box[1][0]
        y1 = box[0][1]
        y2 = box[1][1]

        if point[0] >= x1 and point[0] <= x2 and point[1] >=y1 and point[1]<= y2:
            return True
        return False


    def drawBox(self,box):
        cv2.rectangle(self.origImage,tuple(box[0]),tuple(box[1]),(255,255,255), 4)


    def isLineOutsideBox(self,box, line):
        x1 = box[0][0]
        x2 = box[1][0]
        y1 = box[0][1]
        y2 = box[1][1]
        if line[0] >= x1 and line[2] <= x2:
            if line[1] >= y1 and line[3]<=y2:
                return False
        return True


    def removeFromBoxOfDoom(self,lines):
        box = self.getBoxOfDoom()
        linesNew =[]
        linesNew = list(filter(lambda line:self.isLineOutsideBox(box,line),lines))
        return linesNew


    def line(self,line):
        A = (line[1] - line[3])
        B = (line[2] - line[0])
        C = (line[0]*line[3] - line[2]*line[1])
        return A, B, -C


    def lineIntersection(self,L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False


    def isPointInImage(self,point):
        if point[0] <0 or point[1] <0 or point[0]>self.rows or \
            point[1]>self.cols or not self.isPointValid(point):
            return False
        return True

    # Find intersections between multiple lines (not line segments!)
    def findIntersections(self,lines):
        intersections = []
        lines = list(map(self.line, lines))
        # lines = list(filter(lambda l : l[0]/l[1] <0, lines))
        for i, line_1 in enumerate(lines):
            for line_2 in lines[i + 1:]:
                intersection = self.lineIntersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)
        return intersections

    def drawIntersections(self,points):
        for point in points:
            point = tuple([int(point[0]), int(point[1])])
            cv2.circle(self.origImage,tuple(point), 2,(0,0,255), -11)


    def run(self):
        lines = self.drawLineSegments()
        intersections = self.findIntersections(lines)
        intersections = list(filter(lambda point: self.isPointInImage(point),intersections))
        self.drawIntersections(intersections)
        best_cell = self.findBestCell(intersections,100)
        self.visualizePath(best_cell)

        cv2.imshow("img", self.origImage)



def main():


    Vp = GetVanishingPoint()


    for i in range(1,100):
        file = os.path.join(folderPath,files[i])
        image = cv2.imread(file)
        Vp.setImage(image)
        Vp.run()
        cv2.waitKey(delay = 0)


if __name__ == "__main__":
    main()