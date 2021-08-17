# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:38:39 2020

@author: VIVEK RUGLE
"""

import numpy as np
import cv2
import imutils #provides img processing tools  eg. resize, rotn
import os # functions for interacting with os
from scipy.spatial import distance as dist # computes distance between two 1D arrays

MIN_CONF = 0.3
NMS_THRESH = 0.3

def detect_people(frame, net, ln, personIdx=0):
	# grab the dimensions of the frame and  initialize the list of results
	(H, W) = frame.shape[:2]
	results = []

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False) #(img,scalefactor,size,colour swap)
	net.setInput(blob) #img blob passed to yolo object detector model
	layerOutputs = net.forward(ln) 

	# initialize our lists of detected bounding boxes, centroids, and confidences, respectively
	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

            #Updating the results list with person prediction prob, bbox coordinates & centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return results



# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolov3.weights"])
configPath = os.path.sep.join(["yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #DNN library allows to load pre-trained networks

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing the video ...")
vs = cv2.VideoCapture("peoples.mp4")
writer = None

MIN_DISTANCE = 40 # Minimum distance in pixels

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
    
	if not grabbed: #Reached the end
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln, personIdx=LABELS.index("person")) #net=trained model, ln=output layers from yolo

    
    # initialize the set of indexes that violate the minimum social distance
	violate = set()

	# ensure there are *at least* two people detections
	if len(results) >= 2:
		# extract all centroids from the results and compute the Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
                
				if D[i, j] < MIN_DISTANCE:
					# update our violation set with the indexes of the centroid pairs
					violate.add(i)
					violate.add(j)


	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		
		(startX, startY, endX, endY) = bbox #Bounding box coordinates
		(cX, cY) = centroid #Centroid coordinates
		color = (0, 255, 0)
    
        # if the index pair exists within the violation set, then update the color
		if i in violate:
			color = (0, 0, 255) #RED
        
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) #Bbox
		cv2.circle(frame, (cX, cY), 5, color, 1) #Centroid

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"): #Press Q to end
		break