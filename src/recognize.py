# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import cPickle
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=False,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=False, 
	help="path to the tesitng images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


if args["training"]:
	# loop over the training images
	for imagePath in paths.list_images(args["training"]):
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
	
		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split("/")[-2])
		data.append(hist)
	
	# train a Linear SVM on the data
	model = LinearSVC(C=100.0, random_state=42)
	model.fit(data, labels)
	f = open("model.cpickle", "w")
	f.write(cPickle.dumps(model))
	f.close()


elif args["testing"]:
	model = cPickle.loads(open("model.cpickle").read())
	# loop over the testing images
	for imagePath in paths.list_images(args["testing"]):
		print("procesando ... " + imagePath)

		# load the image, convert it to grayscale, describe it,
		# and classify it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		prediction = model.predict(hist)[0]
		
		print(imagePath+ " - " + prediction)
		
		newName = imagePath.split('.')		
		os.rename(imagePath, newName[0]+'-'+prediction+'.'+newName[1])
		
	
		# display the image and the prediction
# 		cv2.putText(image, prediction, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
# 			5.0, (0, 0, 255), 3)
		
		# window
# 		screen_res = 1280, 720
# 		scale_width = screen_res[0] / image.shape[1]
# 		scale_height = screen_res[1] / image.shape[0]
# 		scale = min(scale_width, scale_height)
# 		window_width = int(image.shape[1] * scale)
# 		window_height = int(image.shape[0] * scale)
# 	
# 		cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
# 		cv2.resizeWindow('dst_rt', window_width, window_height)
# 		
# 		cv2.imshow('dst_rt', image)
# 		cv2.waitKey(0)