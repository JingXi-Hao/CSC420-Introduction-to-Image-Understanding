import imutils
import pickle
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def extract_embeddings_and_labels(input_path):
	# initialize variables and file paths
	#input_path = "../training_set/"
	races = ["white", "black", "asian", "indian", "others"]
	embedding_model_path = "openface_nn4.small2.v1.t7"

	# define results going to be returned
	true_labels = []
	images_embeddings = []

	# initialize face embedding model
	embedding_model = cv2.dnn.readNetFromTorch(embedding_model_path)

	# loop over each training images to extract its embedding
	for i in range(len(races)):
		race = races[i]
		race_path = input_path + races[i] + "/"
		images_list = os.listdir(race_path)

		for img_name in images_list:
			if img_name != ".DS_Store":

				# read the image and get its width and height
				img_path = race_path + img_name
				img = cv2.imread(img_path)
				h = img.shape[0]
				w = img.shape[1]

				# extract the true label for the image
				labels = img_name.split("_")
				race_idx = int(labels[0])

				# preprocess the image and construct a blob for each face 
				# image and pass it face embedding model and get a 128-d 
				# embedding vector for the face image
				img_blur = cv2.GaussianBlur(img, (5,5), 0)
				face_blob = cv2.dnn.blobFromImage(img, 1.0/255, (96, 96), (0, 0, 0),
					swapRB=True, crop=False)
				embedding_model.setInput(face_blob)
				embedding_vector = embedding_model.forward()

				# append the embedding vector and true label to corresponding lists
				images_embeddings.append(embedding_vector.flatten())
				true_labels.append(race_idx)

	# store the image_embeddings and true_labels using pickle
	data_extracted = {"embeddings": images_embeddings, "labels": true_labels}	
	pickle.dump(data_extracted, open("training_set_embeddings_and_labels.p", "wb"))

	return (images_embeddings, true_labels)

def train_race_recognition_model(embeddings, true_labels):
	# initialize label encoder and encode the true labels
	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(true_labels)

	# initialize face recognition model and train it use the embeddings matrix
	race_recognition_model = SVC(C=1.0, kernel="linear", gamma="scale", probability=True)
	race_recognition_model.fit(embeddings, labels)

	# store the trained face recognition model and label encoder
	data_needed = {"model": race_recognition_model, "encoder": label_encoder}
	pickle.dump(data_needed, open("model_and_encoder.p", "wb"))

	return (race_recognition_model, label_encoder)

def test_accuracy(race_recognition_model, label_encoder, test_embeddings, test_labels):
	count = 0

	for i in range(len(test_embeddings)):
		embedding_vector = test_embeddings[i]
		true_label = test_labels[i]
		predictions = race_recognition_model.predict_proba(np.array(embedding_vector).reshape(1, -1))[0]
		j = np.argmax(predictions)
		max_probability = predictions[j]
		predicted_race = label_encoder.classes_[j]

		if true_label == predicted_race:
			count += 1

	acc = float(count) / len(test_embeddings)

	return acc

def recognize_race(img, face, face_aligned, race_recognition_model, label_encoder):
	# define embedding model
	embedding_model_path = "openface_nn4.small2.v1.t7"
	embedding_model = cv2.dnn.readNetFromTorch(embedding_model_path)

	# define bounding box position
	(left, top, right, bottom) = face.astype("int")

	# use aligned face to extract embedding vector and pass it into the
	# trained race recognition model
	face_aligned_blur = cv2.GaussianBlur(face_aligned, (5,5), 0)
	face_blob = cv2.dnn.blobFromImage(face_aligned_blur, 1.0/255, (96, 96), (0, 0, 0),
			swapRB=True, crop=False)
	embedding_model.setInput(face_blob)
	embedding_vector = embedding_model.forward()
	predictions = race_recognition_model.predict_proba(embedding_vector)[0]
	j = np.argmax(predictions)
	max_probability = predictions[j]
	predicted_race = label_encoder.classes_[j]

	# draw the bounding box of the face along with the associated probability
	text = "{}: {:.2f}%".format(predicted_race, max_probability * 100)
	if top - 10 > 10: # adjust text position if the face detected occurs at the top of image
		text_y = top - 10
	else:
		text_y = top + 10  
	cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
	cv2.putText(img, text, (left, text_y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
