import cv2 as cv
import numpy as np
import math
import sys
import os
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt 
import pickle

sys.path.append("tensorflow/models/research/")

import object_detection.utils.label_map_util as label_map_util

import object_detection.utils.visualization_utils as vis_util

# question 2(a)
# helper function for 2(a) to read in camera information
def read_camera_info(calib_path):
    # index 0 --> f; index 1 --> px; index 2 --> py; index 3
    # --> baseline
    camera_info = [0, 0, 0, 0]
        
    file = open(calib_path, "r")
    
    for line in file:
        token_list = line.split(": ")
        if token_list[0] ==  'f':
            camera_info[0] = float(token_list[1].strip('\n'))
        elif token_list[0] == 'px':
            camera_info[1] = float(token_list[1].strip('\n'))
        elif token_list[0] == 'py':
            camera_info[2] = float(token_list[1].strip('\n'))
        elif token_list[0] == 'baseline':
            camera_info[3] = float(token_list[1].strip('\n'))
    # close the file
    file.close()
    
    return camera_info
    
# helper funtion for 2(a) to compute the depth matrix
def compute_depth_matrix(disparity_path, camera_info):
    left_disparity = cv.imread(disparity_path, 0)
    #print(left_disparity)
    row, column = left_disparity.shape
    result = np.zeros((row, column))
    
    for i in range(row):
        for j in range(column):
            if left_disparity[i][j] != 0:
                result[i][j] = (camera_info[0] * camera_info[3]) / (left_disparity[i][j])
            else:
                result[i][j] = float('inf')
        
    return result

# question 2(a) - find depth for each pixel
def compute_depth_for_image(image_numbers, test_results_filename, test_calib_filename):
    depth_list = []
    camera_info_list = []
    
    for i in range(len(image_numbers)):
        # get the image number
        image_number = image_numbers[i]
        
        # read camera information and store all those information
        calib_path = '{}/{}_allcalib.txt'.format(test_calib_filename, image_number)
        camera_info = read_camera_info(calib_path)
        camera_info_list.append(camera_info)
        
        # read the left disparity image
        disparity_path = '{}/{}_left_disparity.png'.format(test_results_filename, image_number)
        result = compute_depth_matrix(disparity_path, camera_info)
        
        depth_list.append(result)
    
    return (depth_list, camera_info_list)
    
# question 2(b) - question 2(f)
# helper function for question 2 to load a (frozen) Tensorflow model into memory
def load_tensorflow_model():
    MODEL_NAME = 'tensorflow/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    return (detection_graph, category_index)

# helper function for question 2 to load image into numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# helper function for question 2 to run inference for single image
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
  
# helper fuction for 2(b) to deal with detection information
def extract_detection_information(threshold, total_detections, detection_boxes, detection_classes, detection_scores):
    result_dict = {}
    result_dict['detection_boxes'] = []
    result_dict['detection_scores'] = []
    result_dict['detection_classes'] = []
    count = 0
    needed_classes = [1, 2, 3, 10]
    
    for i in range(total_detections):
        if (detection_scores[i] >= threshold) and (detection_classes[i] in needed_classes):
            score = detection_scores[i]
            box = detection_boxes[i]
            single_class = detection_classes[i]
            
            result_dict['detection_boxes'].append(box)
            result_dict['detection_scores'].append(score)
            result_dict['detection_classes'].append(single_class)
            
            count = count + 1
            
    result_dict['num_detections'] = count
    
    return result_dict
    
  
# question 2(b) - get detection information and store them
def get_and_store_detection_info(image_numbers, detection_graph, category_index, threshold):
    results = []
    for k in range(len(image_numbers)):
        image_path = "./data/test/left/{}.jpg".format(image_numbers[k])
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # extract needed detection information
        total_detections = output_dict['num_detections']
        detection_boxes = output_dict['detection_boxes']
        detection_classes = output_dict['detection_classes']
        detection_scores = output_dict['detection_scores']
        
        result = extract_detection_information(threshold, total_detections, detection_boxes, detection_classes, detection_scores)
        
        results.append(result)
            
        pickle.dump(result, open( './results/{}_detection_information.p'.format(image_numbers[k]), "wb" ))
        #try_read = pickle.load( open( './results/{}_detection_information.p'.format(image_numbers[k]), "rb" ))
        #print(try_read)
        
    return results
        
# helper function for 2(c) to set bounding box color
def set_color(detection_class):
    color = (0, 0, 255)
    
    if detection_class == 1:
        color = (255, 0, 0)
    elif detection_class == 2:
        color = (0, 255, 0)
    elif detection_class == 3:
        color = (0, 0, 255)
    elif detection_class == 10:
        color = (0, 255, 255)
        
    return color

# question 2(c) - draw box around object recognized
def draw_bounding_boxes(detection_results, image_numbers):
    category_index = load_tensorflow_model()[1]
    
    for i in range(len(image_numbers)):
        image_path = "./data/test/left/{}.jpg".format(image_numbers[i])
        image = cv.imread(image_path).copy()
        height = image.shape[0]
        width = image.shape[1]
        detection = detection_results[i]
        total_detections = detection['num_detections']
        
        for j in range(total_detections):
            detection_box = detection['detection_boxes'][j]
            detection_class = detection['detection_classes'][j]
            detection_score = detection['detection_scores'][j]
            
            top = int(detection_box[0] * height)
            left = int(detection_box[1] * width) 
            bottom = int(detection_box[2] * height)
            right = int(detection_box[3] * width)
            
            # set color for bounding box
            color = set_color(detection_class)
            
            # draw box to show the object recognition
            cv.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # add label on bounding box
            label_top = top
            if top < 25:
                label_top = bottom + 15
            category_label = category_index[detection_class]['name']
            cv.putText(image, category_label, (left, label_top), cv.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
            
            # wirte the image in order to stor it
            cv.imwrite('./output/{}_with_box.jpg'.format(image_numbers[i]), image)
            
# helper function for question 2(d) to compute mass for a single object detected
def compute_object_mass(depth_matrix, image, top, left, bottom, right):
    total_row = 0
    total_column = 0
    total_depth = 0
    #print(depth_matrix)
    
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            #print(i, j)
            depth = depth_matrix[i][j]
            if depth != float('inf'):
                total_row = total_row + depth * (i)
                total_column = total_column + depth * (j)
                total_depth  = total_depth + depth
            else:
                total_row = total_row + sys.maxint * (i)
                total_column = total_column + sys.maxint * (j)
                total_depth  = total_depth + sys.maxint
                
    result_row = int(total_row / total_depth)
    result_column = int(total_column / total_depth)
    result_z = depth_matrix[result_row][result_column]
            
    return (result_row, result_column)

            
# question 2(d) - find the central mass for each object detected
def compute_center_of_mass(image_numbers, detection_results, depth_list):
    masses = []
    # loop over each image
    for i in range(len(image_numbers)):
        image_path = "./data/test/left/{}.jpg".format(image_numbers[i])
        image = cv.imread(image_path).copy()
        height = image.shape[0]
        width = image.shape[1]
        detection = detection_results[i]
        total_detections = detection['num_detections']
        depth_matrix = depth_list[i]
        
        # open the file
        file = open('./output/{}_central_mass_info.txt'.format(image_numbers[i]), 'w')
        
        mass_list = []
        
        # loop over each detected object
        for j in range(total_detections):
            detection_box = detection['detection_boxes'][j]
            detection_class = detection['detection_classes'][j]
            detection_score = detection['detection_scores'][j]
            
            top = int(detection_box[0] * height)
            left = int(detection_box[1] * width) 
            bottom = int(detection_box[2] * height)
            right = int(detection_box[3] * width)
            
            mass = compute_object_mass(depth_matrix, image, top, left, bottom, right)
            coordinate = (top, left, bottom, right)
            
            file.write('{}\n'.format(str(mass)))
            cv.circle(image, (mass[1], mass[0]), 5, (255, 255, 0), -1)
            mass_list.append([coordinate, mass])
        
        # close the file
        file.close()
        cv.imwrite('./output/{}_central_mass_circle.jpg'.format(image_numbers[i]), image)
        masses.append(mass_list)
    return masses
        
# question 2(e) - find the segmentation
def find_segmentation(image_numbers, masses, depth_list, camera_info_list):
    
    for i in range(len(image_numbers)):
        image_path = "./data/test/left/{}.jpg".format(image_numbers[i])
        image = cv.imread(image_path).copy()
        height = image.shape[0]
        width = image.shape[1]
        result_image = np.zeros((height, width))
        
        box_mass_list = masses[i]
        depth_matrix = depth_list[i]
        camera_info = camera_info_list[i]
        
        for j in range(len(box_mass_list)):
            box_coor = box_mass_list[j][0]
            mass = box_mass_list[j][1]
            mass_row = mass[0]
            mass_column = mass[1]
            
            top = box_coor[0]
            left = box_coor[1]
            bottom = box_coor[2]
            right = box_coor[3]
            
            depth = depth_matrix[mass_row][mass_column]
            
            center_row = (mass_row - camera_info[2]) * depth / camera_info[0]
            center_column = (mass_column - camera_info[1]) * depth / camera_info[0]
            
            for k in range(top, bottom + 1):
                for l in range(left, right + 1):
                    z_cor = depth_matrix[k][l]
                    y_cor = (k - camera_info[2]) * z_cor / camera_info[0]
                    x_cor = (l - camera_info[1]) * z_cor / camera_info[0]
                    
                    coor1 = np.array([x_cor, y_cor, z_cor])
                    coor2 = np.array([center_column, center_row, depth]) 
                    distance = np.linalg.norm(coor1 - coor2)
                    
                    if abs(distance) <= 3:
                        result_image[k][l] = 255 - (j*20)
        cv.imwrite("./output/{}_segmentation.jpg".format(image_numbers[i]), result_image)
        
# question 2(f) - create a textual description of the scene
def make_description(image_numbers, masses, depth_list, detection_results, camera_info_list):
    category_index = load_tensorflow_model()[1]
    
    for i in range(len(image_numbers)):
        box_mass_list = masses[i]
        depth_matrix = depth_list[i]
        detection = detection_results[i]
        camera_info = camera_info_list[i]
        
        #total_detections = detection['num_detections']
        detection_boxes = detection['detection_boxes']
        detection_classes = detection['detection_classes']
        detection_scores = detection['detection_scores']
        
        # get number of each classes
        num_persons = detection_classes.count(1)
        num_bicycles = detection_classes.count(2)
        num_cars = detection_classes.count(3)
        num_lights = detection_classes.count(10)
        
        min_distance = float('inf')
        min_index = 0
        min_center_column = 0
        min_center_row = 0
        
        # find the closest object index
        for j in range(len(box_mass_list)):
            box_cor = box_mass_list[j][0]
            mass = box_mass_list[j][1]
            mass_row = mass[0]
            mass_column = mass[1]
            
            top = box_cor[0]
            left = box_cor[1]
            bottom = box_cor[2]
            right = box_cor[3]
            
            depth = depth_matrix[mass_row][mass_column]
            
            center_row = (mass_row - camera_info[2]) * depth / camera_info[0]
            center_column = (mass_column - camera_info[1]) * depth / camera_info[0]
            
            distance = np.linalg.norm(np.array([center_column, center_row, depth]))
            
            if abs(distance) < min_distance:
                min_distance = distance
                min_index = j
                min_center_column = center_column
                min_center_row = center_row
                
        # find label for object
        min_class = detection_classes[min_index]
        min_label = category_index[min_class]['name']
        
        # find where the object is ---> left or right
        if min_center_column >= 0:
            position = 'to your right'
        else:
            position = 'to your left'
            
        print('In the image of {}.jpg, there are {} people, {} bicycles, {} cars, and {} traffic lights.\n'.format(image_numbers[i], num_persons, num_bicycles, num_cars, num_lights))
        print('There is a {} {:.1f} meters {} \n'.format(min_label, abs(min_center_column), position))
        print('It is {:.1f} meters away from you \n\n'.format(min_distance))
        
# define the main function here
if __name__ == "__main__":
    # question 2(a)
    image_numbers = ['004945', '004964', '005002']
    test_results_filename = "./data/test/results"
    test_calib_filename = "./data/test/calib"
    depth_list, camera_info_list = compute_depth_for_image(image_numbers, test_results_filename, test_calib_filename)
    
    for i in range(len(depth_list)):
        cv.imwrite('./output/{}_depth_matrix.jpg'.format(image_numbers[i]), depth_list[i])
        print('The depth matrix for image {}.jpg is {} \n'.format(image_numbers[i], depth_list[i]))
        
    # question 2(b)
    threshold = 0.35
    detection_graph, category_index = load_tensorflow_model()
    detection_results = get_and_store_detection_info(image_numbers, detection_graph, category_index, threshold)
        
    # question 2(c)
    draw_bounding_boxes(detection_results, image_numbers)
    
    # questino 2(d)
    masses = compute_center_of_mass(image_numbers, detection_results, depth_list)
    
    # question 2(e)
    find_segmentation(image_numbers, masses, depth_list, camera_info_list)
    
    # question 2(f)
    make_description(image_numbers, masses, depth_list, detection_results, camera_info_list)

    
