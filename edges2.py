# import the necessary packages
import argparse
import cv2
import imutils
import numpy as np

def getOrientedEdges(img, dx=1, dy=0):
	# compute the Scharr gradient representation of the blackhat image and scale the
	# resulting image into the range [0, 255]
	grad = cv2.Sobel(img, ddepth=cv2.cv.CV_32F, dx=dx, dy=dy, ksize=-1)
	grad = np.absolute(grad)
	(minVal, maxVal) = (np.min(grad), np.max(grad))
	grad = (255 * ((grad - minVal) / (maxVal - minVal))).astype("uint8")
	cv2.imshow("Grad", grad)

	# getting thicker edges
	#grad = cv2.GaussianBlur(grad, (3, 3), 0)

	k_length = 3
	kernelSize = (1,k_length)
	if dy>0:
		kernelSize = (k_length,1)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
	print(grad.dtype)

	thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#cv2.imshow("Thresh", thresh)

	k_length = 3
	kernelSize =(1,k_length)
	if dy>0:
		kernelSize = (k_length,1)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)

	# erode, or erode + dillation
	final = cv2.erode(thresh, kernel, iterations=1)
	#final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 
	#cv2.imshow("Eroded", eroded)

	# can I thin edges more? have to implement skeleton function

	return final

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
ar = float(image.shape[1])/image.shape[0]
canonical_width = 1024
image = cv2.resize(image, (int(canonical_width), int(canonical_width/ar)))

# grab the dimensions of the image and calculate the center of the image
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)
# rotate our image by 45 degrees
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotateda = cv2.warpAffine(image, M, (w, h))
#cv2.imshow("Rotated by 45 Degrees", rotateda)

gray = cv2.cvtColor(rotateda, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow("Image", blurred)

equalized = cv2.equalizeHist(gray)
cv2.imshow("Image eq", equalized)


#kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#sharpen_1 = cv2.filter2D(blurred, -1, kernel_sharpen_1)
#cv2.imshow("Image sharpen", sharpen_1)



gradX = getOrientedEdges(equalized, dx=1, dy=0)

gradY = getOrientedEdges(equalized, dx=0, dy=1)

cv2.imshow("GradY", gradY)
cv2.imshow("GradX", gradX)

edges = cv2.bitwise_or(gradX, gradY)
cv2.imshow("Edges", edges)

kernelSize = (2,2)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernelSize)
final = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

kernelSize = (5,5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imshow("final", final)

# show the images
cv2.waitKey(0)