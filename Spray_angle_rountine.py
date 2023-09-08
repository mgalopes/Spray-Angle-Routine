# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:36:14 2022

@author: Peixe
"""
import cv2
import math

path = '/home/mglopes/Documents/UHPDI/frente/Test1/inferior_1_72/Img000751.tif'
img = cv2.imread(path)
pointsList = []

# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply thresholding to convert grayscale to binary image
ret,thresh = cv2.threshold(gray,6,255,cv2.THRESH_BINARY)

imgRGB = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
 
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(imgRGB,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(imgRGB,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
 
def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
 
def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angD = round(math.degrees(angR),3)
    cv2.putText(imgRGB,str(angD),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
 
 
while True:
    if len(pointsList) % 3 == 0 and len(pointsList) !=0:
        getAngle(pointsList)
 
 
    cv2.imshow('Image',imgRGB)
    cv2.setMouseCallback('Image',mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        pointsList = []
        
    elif cv2.waitKey(1)  == ord('q'):
        cv2.destroyAllWindows()
        exit(1)
    
