import cv2
import numpy as np
import os

os.chdir(r"d:\University Files\Semester-7\IN4640 - Machine Vision\Assignment-2\Materials")

# 1. Load the image
img = cv2.imread("earrings.jpg")
if img is None:
    raise FileNotFoundError("Image not found.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image. The background is white (close to 255), earrings are dark
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area to find the largest ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Keep the two largest contours (assuming there are two earrings)
if len(contours) >= 2:
    earring_contours = contours[:2]
else:
    earring_contours = contours

# 2. Camera parameters
f_mm = 8.0              # Focal length (mm)
Z_mm = 720.0            # Distance to object (mm)
pixel_size_um = 2.2     # Pixel size (um)
pixel_size_mm = pixel_size_um / 1000.0  # Pixel size (mm)

# Magnification M = f / Z
# Object size X = x * Z / f  where x is sensor size in mm.
# Also x = pixels * pixel_size_mm
# So X = (pixels * pixel_size_mm) * Z / f

print("Camera Parameters:")
print(f"Focal length (f): {f_mm} mm")
print(f"Distance to object (Z): {Z_mm} mm")
print(f"Pixel size: {pixel_size_um} um ({pixel_size_mm} mm)")
print(f"Magnification (f/Z): {f_mm/Z_mm:.4f}\n")

print("Calculated Sizes:")
for i, contour in enumerate(earring_contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # physical size in mm
    width_mm = (w * pixel_size_mm) * Z_mm / f_mm
    height_mm = (h * pixel_size_mm) * Z_mm / f_mm
    
    print(f"Earring {i+1}:")
    print(f"  Pixel dimensions: {w} x {h} pixels")
    print(f"  Physical dimensions: {width_mm:.2f} mm x {height_mm:.2f} mm")
    
    # Draw bounding box on the original image for verification (optional)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"E{i+1}: {width_mm:.1f}x{height_mm:.1f}mm", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save annotated image just in case
cv2.imwrite("earrings_annotated.jpg", img)
