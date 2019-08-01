import numpy as np
import cv2
from collections import OrderedDict
import dlib

def detect_faces(input_img_path):
	# initialize variables and file paths
	#input_img_path = "../test_photos/group_photo4.jpg"
	prototxt_path = "deploy.prototxt.txt"
	model_path = "res10_300x300_ssd_iter_140000.caffemodel"
	confidence_threshold = 0.5
	result = []

	# load serialized model from disk
	neural_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
	 
	# load and resize input image to 300x300 pixels, then normalize it to construct input blob 
	img = cv2.imread(input_img_path)
	(h, w) = (img.shape[0], img.shape[1])
	img_blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass blob through the neural network to obtain possible face detections
	neural_net.setInput(img_blob)
	face_detections = neural_net.forward()

	# loop over the possible detections and filter out weak ones with confidence less than threshold,
	# then draw a bounding box around each good detections
	for i in range(0, face_detections.shape[2]):
		confidence = face_detections[0, 0, i, 2]
	 
		# draw a bounding box around strong detections with confidence greater than the threshold
		if confidence > confidence_threshold:
			# compute the coordinates of the bounding box
			box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			result.append(box)
			(left, top, right, bottom) = box.astype("int")
	 
			# draw the bounding box of the face along with the associated probability
			text = "{:.2f}%".format(confidence * 100)
			if top - 10 > 10: # adjust text position if the face detected occurs at the top of image
				text_y = top - 10
			else:
				text_y = top + 10  
			cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
			cv2.putText(img, text, (left, text_y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	 
	# show the output image
	cv2.imwrite("detection.jpg", img)

	return result

def find_landmarks(input_img_path, face):
	# initialize the facial landmark predictor
	shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
	predictor = dlib.shape_predictor(shape_predictor_path)

	# initialize variables
	img = cv2.imread(input_img_path)
	gray_img = cv2.imread(input_img_path, 0)

	# use predictor to find the facial landmarks
	(left, top, right, bottom) = face.astype("int")
	rect = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
	landmarks = predictor(gray_img, rect)

	# initialize the list of 68 (x, y)-coordinates used for dlib’s 68-point
	# facial landmark 
	coords = np.zeros((landmarks.num_parts, 2), dtype="int")

	# find the actual coordinates of the 68 facial landmarks on the detected 
	# face and put in coords
	for j in range(0, landmarks.num_parts):
		coords[j] = (landmarks.part(j).x, landmarks.part(j).y)

	return coords

def align_face(input_img_path, face):
	# define an ordered dictionary for dlib’s 68-point facial landmark detector 
	# that maps the indexes of facial landmarks to specific face regions
	FACIAL_LANDMARKS_68_IDXS = OrderedDict([
		("mouth", (48, 68)),
		("right_eyebrow", (17, 22)),
		("left_eyebrow", (22, 27)),
		("right_eye", (36, 42)),
		("left_eye", (42, 48)),
		("nose", (27, 36)),
		("jaw", (0, 17))
	])

	# extract facial landmark positions
	landmarks = find_landmarks(input_img_path, face)

	# initialize the ratio of the desired eyes position and the desired face size 
	left_eye = 0.35
	right_eye = 1.0 - left_eye
	face_size = 100

	# extract the coordinates of the left eye and right eye 
	(left_start, left_end) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(right_start, right_end) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
	left_eye_pts = landmarks[left_start:left_end]
	right_eye_pts = landmarks[right_start:right_end]

	# compute the center of mass for each eye in each detection
	left_eye_com = left_eye_pts.mean(axis=0).astype("int")
	right_eye_com = right_eye_pts.mean(axis=0).astype("int")

	# compute the angle between the center of mass for each eye
	delta = right_eye_com - left_eye_com
	angle = np.degrees(np.arctan2(delta[1], delta[0])) - 180

	# calculate scale between the desired distance and the actual distance 
	# of the eyes
	actual_dist = np.sqrt(np.sum(delta ** 2))
	desired_dist = (right_eye - left_eye) * face_size
	scale = desired_dist / actual_dist

	# compute the coordinates of the midpoint between the two eyes in the
	# input image
	midpoint = tuple((left_eye_com + right_eye_com) // 2)

	# calculate the 2D affine transformation matrix 
	M = cv2.getRotationMatrix2D(midpoint, angle, scale)

	# update the translation component of the matrix
	M[0, 2] += face_size / 2.0 - midpoint[0]
	M[1, 2] += face_size * left_eye - midpoint[1]

	# apply affine transformation on img using M and returned the aligned face
	img = cv2.imread(input_img_path)
	aligned_face = cv2.warpAffine(img, M, (face_size, face_size), 
									flags=cv2.INTER_CUBIC)

	return aligned_face

