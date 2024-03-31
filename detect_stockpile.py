import numpy as np
import argparse
import time
import cv2
import os
from  PIL import Image
from tqdm import tqdm

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def generate_crops(image_file, winW=512, winH=512, stepSize=512 // 2):
    image = cv2.imread(image_file)
    height = np.size(image, 0)
    width = np.size(image, 1)

    crops = []
    resized = image
    if width / winW > 5 or width / winH > 5:
        print("Warning! Huge image, may take 5 mins")
        winW, winH, stepSize = winW, winH, stepSize
    for (x, y, window) in sliding_window(resized, stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        crops.append([x, y, x + winW, y + winH])
    return crops


def non_max_suppression(boxes, overlapThresh=0.4):
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return []
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick]

def yolo_predict(image, confidence=.01, threshold=.3, weightsPath="G:\\darknet\\backup\\tinyyolo_best.weights", configPath="G:\\darknet\\cfg\\tinyyolo.cfg", labelsPath = 'G:\\darknet\\data\\obj.names'):
	bbox = []
	# image = cv2.imread(image)
	(H, W) = image.shape[:2]
	LABELS = open(labelsPath).read().strip().split("\n")
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (800, 800),swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			conf = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if conf > confidence:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(conf))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence,threshold)

# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			bbox.append([boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]])

	return bbox
def predict_stockpile(image_path):
	final_box = []
	crops = generate_crops(image_path, winW=1000, winH=1000, stepSize=500)
	img = cv2.imread(image_path)
	draw_temp = img.copy()
	start = time.time()
	for crop in tqdm(crops):
		img_ = img[crop[1]:crop[3], crop[0]:crop[2]]
		
		boxes = yolo_predict(img_)
		if boxes:
			boxes = np.asarray(boxes)
			for box in boxes:
				b = (box.astype(int))
				if len(b) > 0:
					box = np.asarray(b)
					crop = np.asarray(crop)
					x, y, w, h = b[0], b[1], b[2], b[3]
					x_final, y_final = b[0] + crop[0], b[1] + crop[1]
					final_box.append([x_final, y_final, x_final + w, y_final + h])
	end = time.time()
	print('yolo prediction time: ' + str(end - start))
	final_box = non_max_suppression(final_box)
	for box in final_box:
		draw_box(draw_temp, box, color=(0, 0, 255))
		print(box)
	cv2.imwrite('detected_' + str(image_path) , draw_temp)
	return final_box


# print(yolo_predict('crops/9600_1600_10600_2600.jpg'))
predict_stockpile('stockpile.png')