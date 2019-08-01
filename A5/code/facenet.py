from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import model_from_json
import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans

def load_facenet():
    """
    loads a saved pretrained model from a json file
    :return:
    """
    # load json and create model
    json_file = open('FRmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FRmodel = model_from_json(loaded_model_json)

    # load weights into new model
    FRmodel.load_weights("FRmodel.h5")
    print("Loaded model from disk")

    return FRmodel

def img_to_encoding(img1, model):
    """
    returns 128-dimensional face embedding for input image
    :param img1:
    :param model:
    :return:
    """
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def load_dataset():

    if not os.path.exists('saved_faces/'):
        os.makedirs('saved_faces')

    a = np.load('faces.npy')

    for i in range(a.shape[0]):

        img = a[i][..., ::-1]
        img = cv2.resize(img, (96, 96))
        cv2.imwrite("saved_faces/face_image_"+str(i)+".jpg", img)
    
# question 1(c)
def compute_and_store_embeddings():
    # get the pretrained model
    model = load_facenet()
    # get the images
    a = np.load('faces.npy')
    
    embeddings = []
    
    for i in range(a.shape[0]):
        img = cv2.imread("saved_faces/face_image_"+str(i)+".jpg")
        embedding = img_to_encoding(img, model)
        embeddings.append(embedding.reshape(128))
    
    # store embeddings
    pickle.dump(embeddings, open( './output/embeddings.p', "wb" ))
    
    return embeddings

# question 1(e)    
def cluster_embeddings(embeddings):
    # Compute k-means clustering.
    kmeans = KMeans(n_clusters=6, random_state=0).fit(embeddings)
    
    return kmeans
    
# question 1(f)
def save_visual_words(kmeans):
    visual_words = kmeans.cluster_centers_
    
    # save visual_words
    np.save('./output/visual_words.npy', visual_words)
    
    return visual_words

# question 1(f)
def build_inverted_index(kmeans):
    inverted_index = {}
    
    for i, label in enumerate(kmeans.labels_):
        if label not in inverted_index.keys():
            inverted_index[label] = []
        inverted_index[label].append("saved_faces/face_image_"+str(i)+".jpg")
        
    return inverted_index
    
# question 1(h)
def find_matching_images(visual_words, inverted_index):
    # get all input images names
    dir = "../input_faces/"
    inputs = os.listdir(dir)
    if ('.DS_Store' in inputs):
        inputs.remove('.DS_Store')
    
    # get pretrained model
    model = load_facenet()
    
    results = {}
    threshold = 0.8
    
    for img_idx, img_name in enumerate(inputs): 
        img_path = "{}{}".format(dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))
        
        # compute the embedding for img, for each img only have one embedding
        embedding = img_to_encoding(img, model).reshape(128)
        
        # assign visual word to each embedding of img
        min_distance = float('inf')
        word_idx = 0
        
        for index, word in enumerate(visual_words):
            distance = np.linalg.norm(np.array(embedding) - np.array(word))
            if distance < min_distance:
                min_distance = distance
                word_idx = index
        
        # get matching images
        dict_key = "input_faces/"+img_name
        if min_distance > threshold:
            if dict_key not in results.keys():
                results[dict_key] = []
        else:
            potential_imgs = inverted_index[word_idx]
            if dict_key not in results.keys():
                results[dict_key] = []
            results[dict_key] = potential_imgs
            
    for key, value in results.items():
        print("All matching images passed threshold for {} are: \n".format(key))
        if len(value) == 0:
            print("No matching images.\n")
        else: 
            print("{} \n".format(value))
    
    # store the results before returning it
    pickle.dump(results , open( './output/matching_images.p', "wb" ))
    
    return results

if __name__ == "__main__":
    #load_dataset()
    
    # question 1(c)
    embeddings = compute_and_store_embeddings()
    
    # question 1(e)
    kmeans = cluster_embeddings(embeddings)
    
    # question 1(f)
    visual_words = save_visual_words(kmeans)
    inverted_index = build_inverted_index(kmeans)
    
    # quetion 1(h)
    top_matches = find_matching_images(visual_words, inverted_index)
    
