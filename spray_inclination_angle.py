import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

# Load image
path = 'gasolina_div_40C_70bar_binary.png'
img = cv2.imread(path)

# Convert image to grayscale and threshold it to get a binary image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
imgRGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

# Get image dimensions
height, width = img.shape[:2]
vertical_line_x = width // 2  # Center x for vertical reference

# Global lists to store original click points and computed white midpoints
clickPoints = []
refMidpoints = []

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    """Draw a dashed line between two points (works for vertical or horizontal lines)."""
    # If vertical line: x constant, iterate over y
    if pt1[0] == pt2[0]:
        for y in range(pt1[1], pt2[1], dash_length * 2):
            start_point = (pt1[0], y)
            end_point = (pt1[0], min(y + dash_length, pt2[1]))
            cv2.line(img, start_point, end_point, color, thickness)
    # If horizontal line: y constant, iterate over x
    elif pt1[1] == pt2[1]:
        for x in range(pt1[0], pt2[0], dash_length * 2):
            start_point = (x, pt1[1])
            end_point = (min(x + dash_length, pt2[0]), pt1[1])
            cv2.line(img, start_point, end_point, color, thickness)
    else:
        # For non-axis aligned lines, additional logic would be needed.
        cv2.line(img, pt1, pt2, color, thickness)

def get_white_midpoint(y, binary_img):
    """For the given y coordinate, compute the midpoint (x) of white pixels in that row."""
    if y < 0 or y >= binary_img.shape[0]:
        return None
    row = binary_img[y, :]  # get the row
    white_indices = np.where(row == 255)[0]
    if white_indices.size == 0:
        return None
    mid_x = int(np.mean(white_indices))
    return (mid_x, y)

# Draw a dashed vertical reference line in the middle of the image
draw_dashed_line(imgRGB, (vertical_line_x, 0), (vertical_line_x, height), (0, 255, 0), thickness=2, dash_length=10)

def mousePoints(event, x, y, flags, params):
    """When the user clicks, compute the white midpoint along that horizontal row and draw a dashed line."""
    global clickPoints, refMidpoints, imgRGB
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the click point (for reference, if needed)
        clickPoints.append((x, y))
        
        # Compute the white midpoint along the clicked horizontal row from the threshold image.
        midpoint = get_white_midpoint(y, thresh)
        if midpoint is None:
            print(f"No white pixels found in row {y}.")
            return
        
        # Draw a filled circle at the computed midpoint (in blue)
        cv2.circle(imgRGB, midpoint, 5, (255, 0, 0), cv2.FILLED)
        
        # Draw a dashed horizontal line along this row (covering full width)
        draw_dashed_line(imgRGB, (0, y), (width, y), (0, 255, 255), thickness=2, dash_length=10)
        
        # Save the computed reference midpoint
        refMidpoints.append(midpoint)
        
        # If we have two midpoints, draw a line connecting them and calculate the angle.
        if len(refMidpoints) == 2:
            pt1, pt2 = refMidpoints
            cv2.line(imgRGB, pt1, pt2, (255, 0, 0), 2)
            
            # Calculate the angle with respect to vertical.
            # Use dx and dy from the two reference points:
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            if dy == 0:
                angle = 90.0  # Horizontal line: 90° with vertical
            else:
                angle = math.degrees(math.atan(abs(dx/dy)))
            angle = round(angle, 2)
            
            # Display the calculated angle near the first reference midpoint.
            cv2.putText(imgRGB, f'{angle}\u00B0', (pt1[0] + 10, pt1[1] + 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            print(f"Angle between line (through white midpoints) and vertical: {angle}°")

def show_image():
    """Display the current image using matplotlib."""
    plt.imshow(cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mousePoints)

while True:
    cv2.imshow('Image', imgRGB)
    key = cv2.waitKey(1)
    
    if key == ord('r'):
        # Reset: clear points and reinitialize the image (and redraw the vertical reference line)
        clickPoints = []
        refMidpoints = []
        imgRGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        draw_dashed_line(imgRGB, (vertical_line_x, 0), (vertical_line_x, height), (0, 255, 0), thickness=2, dash_length=10)
    elif key == ord('s'):
        show_image()
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break
