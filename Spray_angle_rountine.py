import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

path = 'average_image_agua_dest_div_25_50bar.png'
img = cv2.imread(path)
pointsList = []

# Convert input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply thresholding to convert grayscale to binary image
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Convert binary image to RGB format for visualization
imgRGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(imgRGB, tuple(pointsList[round((size-1)/3)*3]), (x, y), (255, 0, 0), 2)
        cv2.circle(imgRGB, (x, y), 5, (255, 0, 0), cv2.FILLED)
        pointsList.append([x, y])

def gradient(pt1, pt2):
    if (pt2[0] - pt1[0]) == 0:
        return float('inf')  # Avoid division by zero
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

def getAngle(pointsList):
    if len(pointsList) < 3:
        return
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    if (1 + m2 * m1) == 0:
        return 90  # Handle vertical angles
    angR = math.atan(abs((m2 - m1) / (1 + m2 * m1)))
    angD = round(math.degrees(angR), 3)
    cv2.putText(imgRGB, str(angD), (pt1[0] + 10, pt1[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    return angD

# Show image using matplotlib
def show_image():
    plt.imshow(cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mousePoints)

while True:
    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)
    
    cv2.imshow('Image', imgRGB)
    key = cv2.waitKey(1)
    
    if key == ord('r'):
        pointsList = []
        imgRGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)  # Reset image
    elif key == ord('s'):
        show_image()
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
