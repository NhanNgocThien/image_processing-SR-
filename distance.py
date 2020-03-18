from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import random as rng

rng.seed(12345)
font = cv2.FONT_HERSHEY_SIMPLEX

def getMidPoint(x1, y1, x2, y2):
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5) 

def getDistance(x1, y1, x2, y2):
    # Calculate the distance between mid points
    D = dist.euclidean((x1, y1), (x2, y2)) 
    mX, mY = getMidPoint(x1, y1, x2, y2)
    return D, mX, mY

class Object_Property:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def getCoordinates(self):
        # Top left corner
        self.x1 = self.x
        self.y1 = self.y 
        # Bottom left corner
        self.x2 = self.x 
        self.y2 = self.y + self.height
        # Top right corner
        self.x3 = self.x + self.width
        self.y3 = self.y 
        # Bottom right corner
        self.x4 = self.x + self.width
        self.y4 = self.y + self.height

# Load the image, convert to grayscale and blur it
source_img = cv2.imread('input/nuoc-rua-chen.jpeg')
gray_scale = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
gray_scale = cv2.GaussianBlur(gray_scale, (7,7), 0)

# Perform edge detection, dilation + erosion to close gaps between object edges
edge_detection = cv2.Canny(gray_scale, 50, 100)
edge_detection = cv2.dilate(edge_detection, None, iterations=1)
edge_detection = cv2.erode(edge_detection, None, iterations=1)

# Find contours in the edge map
contours, _ = cv2.findContours(edge_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define bounding area, center and radius
required_obj = [[0, 0], [0, 0]]
index = 0
cnts_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)

# Loop over all contours to find the label and the refObj
for i, c in enumerate(contours):
    cnts_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(cnts_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(cnts_poly[i])
    temp_radius = int(radius[i])
    if(temp_radius <= 100):
        continue
    else:
        # Notes: handle cases that there are more than 3 element in the array or only the bottle is found
        if(index >= 2):
            break
        if(temp_radius < 255): # This is the label
            required_obj[0][0] = i
            required_obj[0][1] = temp_radius
        else:
            required_obj[1][0] = i
            required_obj[1][1] = temp_radius
        index += 1

# Get the x,y,w,h of the label
label_prop = boundRect[required_obj[0][0]]
label = Object_Property(label_prop[0], label_prop[1], label_prop[2], label_prop[3])
label.getCoordinates()

h1, h2 = getMidPoint(label.x1, label.y1, label.x2, label.y2)

w1, w2 = getMidPoint(label.x1, label.y1, label.x3, label.y3)

# Get the x,y,w,h of the refObj
obj_prop = boundRect[required_obj[1][0]]
obj = Object_Property(obj_prop[0], obj_prop[1], obj_prop[2], obj_prop[3])
obj.getCoordinates()

h1_obj, h2_obj = getMidPoint(obj.x1, obj.y1, obj.x2, obj.y2)

w1_obj, w2_obj = getMidPoint(obj.x1, obj.y1, obj.x3, obj.y3)

print(obj.x1, obj.y1)
print(obj.x3, obj.y3)
print(w1_obj, w2_obj)

# Draw the rectangle for label and obj
drawing = np.zeros((edge_detection.shape[0], edge_detection.shape[1], 3), dtype=np.uint8)

color = (rng.randint(0,256), rng.randint(0, 256), rng.randint(0,256))

cv2.rectangle(drawing, (label.x1, label.y1), \
    (label.x4, label.y4), color, 2)

cv2.rectangle(drawing, (obj.x1, obj.y1), \
    (obj.x4, obj.y4), color, 2)

# Draw the heigh and width of the label
cv2.putText(drawing, "{:.1f} pixel".format(label.height), (h1 + 10, h2) , font, 0.55, (255, 0, 0), 2)

cv2.putText(drawing, "{:.1f} pixel".format(label.width), (w1, w2 + 20) , font, 0.55, (255, 0, 0), 2)

# Calculate the distance and get mid point of the line
D, mX, mY = getDistance(w1, w2, w1_obj, w2_obj)

# Draw the line
cv2.line(drawing, (w1, w2), (w1_obj, w2_obj), color, 2)
cv2.line(drawing, (h1, h2), (h1_obj, h2_obj), color, 2)
cv2.putText(drawing, "{:.1f} pixel".format(D), (mX, mY - 10) ,
			font, 0.55,(255, 0, 0), 2)

# Output the image
cv2.imshow("Contours", drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()