import cv2
import numpy as np
import math
import os
import csv

# ==== USER CONFIGURATION ====
FOLDER_PATH = '/home/mlopes/Documents/Test1/output_images/grayscale/etanol_conv_40C_50bar/'      # Folder with input images
HEIGHT_PIXEL = 300                 # Row height (in pixels) to measure spray angle
THRESHOLD_VALUE = 10               # Pixel intensity threshold for binarization
# Automatically build CSV filename from the last folder name
folder_name = os.path.basename(os.path.normpath(FOLDER_PATH))
CSV_PATH = f'spray_angles_{folder_name}.csv'
# =============================

pivot_point = None
angle_results = []

def select_pivot_point(img_path):
    """Let user click on a pivot point in the first image."""
    global pivot_point
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    img_disp = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def mouse_callback(event, x, y, flags, param):
        global pivot_point
        if event == cv2.EVENT_LBUTTONDOWN:
            pivot_point = (x, y)
            cv2.circle(img_disp, pivot_point, 5, (0, 0, 255), cv2.FILLED)
            cv2.imshow("Select Pivot Point", img_disp)

    cv2.imshow("Select Pivot Point", img_disp)
    cv2.setMouseCallback("Select Pivot Point", mouse_callback)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q') and pivot_point is not None:
            cv2.destroyAllWindows()
            break

def calculate_angle(pivot, pt_left, pt_right):
    """Calculate angle between the vectors (pivot -> left) and (pivot -> right)."""
    def vector(p1, p2):
        return np.array([p2[0] - p1[0], p2[1] - p1[1]])

    v1 = vector(pivot, pt_left)
    v2 = vector(pivot, pt_right)

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return None
    cos_theta = dot_product / norm_product
    angle_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)
    return round(angle_deg, 2)

def process_image(img_path, height, pivot):
    """Read, binarize image, find left/right edges at the given height, and calculate angle."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Failed to read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    if height < 0 or height >= binary.shape[0]:
        print(f"[!] Height out of bounds: {height}")
        return None

    row = binary[height, :]
    white_pixels = np.where(row == 255)[0]

    if len(white_pixels) < 2:
        print(f"[!] Not enough white pixels found in {img_path}")
        return None

    x_left = white_pixels[0]
    x_right = white_pixels[-1]

    pt_left = (x_left, height)
    pt_right = (x_right, height)

    return calculate_angle(pivot, pt_left, pt_right)

def main():
    global pivot_point
    images = sorted([f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not images:
        print("[!] No images found in the folder.")
        return

    # Let user choose pivot point on first image
    select_pivot_point(os.path.join(FOLDER_PATH, images[0]))

    # Process each image
    for img_name in images:
        path = os.path.join(FOLDER_PATH, img_name)
        angle = process_image(path, HEIGHT_PIXEL, pivot_point)
        if angle is not None:
            angle_results.append([img_name, angle])
        else:
            angle_results.append([img_name, "N/A"])

    # Save results to CSV
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Angle (degrees)'])
        writer.writerows(angle_results)

    print(f"[✓] Processing complete. Results saved to {CSV_PATH}.")

if __name__ == "__main__":
    main()
