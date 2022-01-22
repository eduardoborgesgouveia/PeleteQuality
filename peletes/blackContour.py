import cv2
import numpy as np

# Load image, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('img/targ.jpg')
ds_factor = 0.2
image = cv2.resize(image, (int(image.shape[1] * ds_factor),int(image.shape[0] * ds_factor)), interpolation = cv2.INTER_AREA)
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    # Obtain bounding rectangle to get measurements
    x,y,w,h = cv2.boundingRect(c)


    # Crop and save ROI
    ROI = original[y:y+h, x:x+w]
    #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1

    # Draw the contour and center of the shape on the image
    cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12), 4)
    #cv2.circle(image, (cX, cY), 10, (320, 159, 22), -1) 

cv2.imshow('image.png', image)
cv2.imshow('thresh.png', thresh)
cv2.waitKey(0)