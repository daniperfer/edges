# import the necessary packages
import argparse
import cv2
import imutils
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
ar = float(image.shape[1])/image.shape[0]
canonical_width = 1024
image = cv2.resize(image, (int(canonical_width), int(canonical_width/ar)))




gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

auto = imutils.auto_canny(blurred)


# close + erode ?
kernelSize=(5,5)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernelSize)
diamond = np.array([[0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0], 
					[1, 1, 1, 1, 1],
					[0, 1, 1, 1, 0],
					[0, 0, 1, 0, 0]], dtype=np.uint8)
closing = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, diamond)
cv2.imshow("Closing", closing)

kernelSize =(3,1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
eroded = cv2.erode(closing, kernel, iterations=2)
cv2.imshow("Eroded", eroded)

# find all contours in the image and draw ALL contours on the image
(cnts, _) = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print "Found {} contours".format(len(cnts))


# show the images
cv2.imshow("Original", clone)
cv2.imshow("Auto", auto)
cv2.waitKey(0)