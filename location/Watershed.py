# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# Load the image
img = cv2.imread("barcode.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image to create a binary image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Perform morphological operations to remove noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Apply the watershed algorithm to the binary image
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(opening, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img, markers)

# Create a mask with the barcode region
barcode_mask = np.zeros(img.shape, dtype=np.uint8)
barcode_mask[markers == 2] = 255

# Find the contours of the barcode in the mask
cnts = cv2.findContours(barcode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Draw a rectangle around the barcode
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)

# Display the result
cv2.imshow("Barcode Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
