import cv2
import numpy as np
import os
import urllib.request

# Change directory to the Materials folder where the images are
os.chdir(r"d:\University Files\Semester-7\IN4640 - Machine Vision\Assignment-2\Materials")

flag_filename = "flag.png"

# Download a sample flag if not already present
if not os.path.exists(flag_filename):
    print("Downloading a flag image...")
    # Sri Lanka flag as a sample
    flag_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Flag_of_Sri_Lanka.svg/800px-Flag_of_Sri_Lanka.svg.png"
    urllib.request.urlretrieve(flag_url, flag_filename)

points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Turf Image", img_display)
            
        if len(points) == 4:
            print("\nFour points selected. Press any key to apply homography.")

def main():
    global img_display
    img = cv2.imread("turf.jpg")
    if img is None:
        raise FileNotFoundError("Image not found: turf.jpg")
    
    img_display = img.copy()

    print("Please click on 4 corners of the turf in the image.")
    print("Order should be: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    
    cv2.namedWindow("Turf Image")
    cv2.setMouseCallback("Turf Image", mouse_callback)
    cv2.imshow("Turf Image", img_display)
    
    # Wait until a key is pressed (the user should have clicked 4 points by then)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        # Destination points (the user's clicks)
        pts_dst = np.array(points, dtype=float)
        
        # Open flag image
        flag_img = cv2.imread(flag_filename)
        if flag_img is None:
            raise FileNotFoundError(f"Flag image {flag_filename} not found.")
            
        h, w = flag_img.shape[:2]
        
        # Source points (corners of the flag image)
        pts_src = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=float)
        
        # Calculate Homography
        H, status = cv2.findHomography(pts_src, pts_dst)
        
        # Warp source image to destination based on homography
        temp_flag_warped = cv2.warpPerspective(flag_img, H, (img.shape[1], img.shape[0]))
        
        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dst.astype(int), 255)
        
        # Inverse mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Black out the turf area in original image
        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        
        # Add the warped flag
        result = cv2.add(img_bg, temp_flag_warped)
        
        # Blend to make it look somewhat realistic (overlay on grass)
        blended = cv2.addWeighted(img, 0.4, result, 0.6, 0)
        
        # Show the result
        cv2.imshow("Result", blended)
        print("Press any key to save and exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save output
        cv2.imwrite("turf_with_flag.jpg", blended)
        print("Saved result as turf_with_flag.jpg")
    else:
        print(f"You only selected {len(points)} point(s). You must select 4 points.")

if __name__ == "__main__":
    main()
