import os
import cv2 as cv
import shutil
import math
import random

def process_data():
	'''Read each image in "unprocessed_dataset/aligned&cropped_faces/UTKFace" and sort them
		into different folders first by race then by gender into "dataset/". Then return the 
		count of each race and gender.'''

	# initialize variables
	input_path = "../unprocessed_dataset/aligned&cropped_faces/UTKFace/"
	races = ["white", "black", "asian", "indian", "others"]
	genders = ["male", "female"]
	output_path = "../dataset/"
	count = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]} # race and gender count

	# create a folder for each race and gender if folder does not already exist
	for race in races:
		for gender in genders:
			file_path = "{}{}/{}/".format(output_path, race, gender)
			if not os.path.exists(file_path):
				os.makedirs(file_path)

	# split each image by race and then by gender 
	input_images = os.listdir(input_path)
	for img_name in input_images:
		if img_name != ".DS_Store":
			labels = img_name.split("_")
			# ignore images that are named incorrectly with missing classifications
			if len(labels) == 4: 
				gender_idx = int(labels[1])
				race_idx = int(labels[2])
				count[race_idx][gender_idx] += 1 # update count
				new_name = "{}_{}_{}.jpg".format(race_idx, gender_idx, count[race_idx][gender_idx])
				src = "{}{}".format(input_path, img_name)
				dest = "{}{}/{}/{}".format(output_path, races[race_idx], genders[gender_idx], new_name)
				shutil.copy(src, dest)

	return count

def split_data(total_count):
	'''Read each image in "dataset/" and split them into training, validaiton and test sets.'''

	# initialize variables
	input_path = "../dataset/"
	races = ["white", "black", "asian", "indian", "others"]
	genders = ["male", "female"]
	train_prop = 0.6
	validation_prop = 0.2

	# create brand new folders for training, validation, and test sets 
	for race in races:
		for set_name in ["training_set", "validation_set", "test_set"]:
			file_path = "../{}/{}/".format(set_name, race)
			if os.path.exists(file_path): # remove folders if it exists
				shutil.rmtree(file_path)	
			os.makedirs(file_path) # create new empty folder

	# split data in the dataset folder into training, validation, and test sets
	for race_idx in range(len(races)):
		for gender_idx in range(len(genders)):
			folder_path = "{}{}/{}/".format(input_path, races[race_idx], genders[gender_idx])
			img_names = os.listdir(folder_path)
			random.seed(3)
			random.shuffle(img_names) # shuffle images to guarantee randomness

			# calculate the number of images required in training, and valiadation sets
			count = total_count[race_idx][gender_idx]
			train_count = math.ceil(count * train_prop)
			validation_count = math.ceil(count * validation_prop)

			# split into training set
			for i in range(train_count):
				src = "{}{}".format(folder_path, img_names[i])
				dest = "../training_set/{}/{}".format(races[race_idx], img_names[i])
				shutil.copy(src, dest)

			# split into validation set
			for i in range(train_count, train_count + validation_count):
				src = "{}{}".format(folder_path, img_names[i])
				dest = "../validation_set/{}/{}".format(races[race_idx], img_names[i])
				shutil.copy(src, dest)

			# split into test set
			for i in range(train_count + validation_count, count):
				src = "{}{}".format(folder_path, img_names[i])
				dest = "../test_set/{}/{}".format(races[race_idx], img_names[i])
				shutil.copy(src, dest)
