# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
img = cv2.imread('img/tar_ver_2.jpg')
ds_factor = 1
image = cv2.resize(img, (int(img.shape[1] * ds_factor),int(img.shape[0] * ds_factor)), interpolation = cv2.INTER_AREA)
razao = (7.0/135.0) #7mm dividido por 100 pixels


brightness = 100
contrast = 100
img = np.int16(image)
img = img * (contrast/127+1) - contrast + brightness
img = np.clip(img, 0, 255)
img = np.uint8(img)
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
# image = cv2.imread(args["image"])
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow("Input", shifted)
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
thresh = 255 - thresh
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=int(100*ds_factor),
	labels=thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	minRect = cv2.minAreaRect(c)
	(x, y), (width, height), angle = minRect
	min_rect = np.int0(cv2.boxPoints(minRect))
	cv2.drawContours(image, [min_rect], 0, (0, 255, 0), 2)
	#cv2.rectangle(image, min_rect, (0, 255, 0), 2)
	x_mm = width * razao * (1/ds_factor)
	y_mm = height * razao * (1/ds_factor)
	area = x_mm * y_mm
	#cv2.putText(image, "{} mm??".format(area), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.putText(image, "{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	print("#" + str(label)+ " - {} mm??".format(area))
# show the output image
cv2.imshow("Output", image)
cv2.imwrite("result.png",image)
cv2.waitKey(0)
