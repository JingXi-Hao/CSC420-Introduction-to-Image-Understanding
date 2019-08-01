import cv2
import data
import face as face_module
import model

if __name__ == '__main__':
	# split data set into training, validation, and test set
	total_count = data.process_data()
	data.split_data(total_count)

	# build training, validation, and test set matrix and its
	# correspounding true labels vector
	train_path = "../training_set/"
	train_embeddings, train_labels = model.extract_embeddings_and_labels(train_path)

	#valid_path = "../validation_set/"
	#valid_embeddings, valid_labels = model.extract_embeddings_and_labels(valid_path)

	test_path = "../test_set/"
	test_embeddings, test_labels = model.extract_embeddings_and_labels(test_path)

	# train race recognition model
	race_recognition_model, label_encoder = model.train_race_recognition_model(
		train_embeddings, train_labels)

	# test accuracy on test set
	acc = test_accuracy(race_recognition_model, label_encoder, test_embeddings, test_labels)

	# detect faces
	query_img_path = "../test_photos/group_photo1.jpg"
	query_img = cv2.imread(query_img_path)
	query_img_copy = query_img.copy()
	faces = face_module.detect_faces(query_img_path)

	# align each face
	for i in range(len(faces)):
		face = faces[i]
		aligned_face = face_module.align_face(query_img_path, face)
		(left, top, right, bottom) = face.astype("int")
		original_face = query_img[top:bottom, left:right]

		cv2.imwrite("./face_original_1_{}.jpg".format(i), original_face)
    	cv2.imwrite("./face_aligned_1_{}.jpg".format(i), aligned_face)

		# recognize race for each aligned face
		recognize_race(query_img_copy, face, aligned_face, race_recognition_model, label_encoder)

	cv2.imwrite("detection1.jpg", query_img_copy)
