import cv2
import argparse
import sys
import numpy as np
import os
from yolo_utils import *

def getOutputNames(net):
	layerNames = net.getLayerNames()

	return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess_file(frame, outs, CONF_THRESH):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]	

	classIds = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = float(scores[classId])

			if confidence > CONF_THRESH:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)

				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)

				left = int(center_x - width / 2)
				top = int(center_y - height / 2)

				classIds.append(classId)
				confidences.append(confidence)
				boxes.append([left, top, width, height])

	return boxes, confidences, classIds			

def drawPredictions(frame, classes, classId, conf, left, top, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	label = '%.2f' % conf	

	if classes:
		assert(classId < len(classes))
		label = '%s : %s' % (classes[classId], label)

		labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		top = max(top, labelSize[1])

		cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

CONF_THRESH = 0.5
NMS_THRESH = 0.5
IMG_HEIGHT = 416
IMG_WIDTH = 416
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "yolov3.txt"
OUTPUT_PATH = "test.jpg"

parser = argparse.ArgumentParser()
parser.add_argument("--image", help = "dadi.jpg")
args = parser.parse_args()	

classes = None

with open(CLASSES_PATH, 'rt') as f:
	classes = f.read().strip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

winName = 'Yolo Object Detection'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

if not os.path.isfile(args.image):
	print("Given input image file does not exist")
	sys.exit(1)

img =cv2.imread(args.image)

blob = cv2.dnn.blobFromImage(img, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop = False)
net.setInput(blob)

outs = net.forward(getOutputNames(net))

boxes, confidences, classIds = postprocess_file(img, outs, CONF_THRESH)

indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

for i in indices:
	i = i[0]
	box = boxes[i]
	left = box[0]
	top = box[1]
	width = box[2]
	height  = box[3]

	drawPredictions(img, classes, classIds[i], confidences[i], left, top, left+width, top+height)

cv2.imshow(winName, img)
cv2.imwrite(OUTPUT_PATH, img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()





