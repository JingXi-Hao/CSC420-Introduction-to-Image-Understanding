import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter, gaussian_laplace
from math import ceil, log
from scipy.spatial import distance

# question 1-a; harris method
def compute_harris_method(img):
    # convert the image to its grayscale version
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # blur the image to smooth the image for removing outliers
    blur = cv2.GaussianBlur(gray,(5,5),7)
    
    # compute R value
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    
    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
    
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    
    R = det - 0.05 * np.multiply(trace,trace)
    
    return R

# question 1-a; brown (harmonic mean) method
def compute_brown_method(img):
    # convert the image to its grayscale version:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # blur the image to smooth the image for removing outliers
    blur = cv2.GaussianBlur(gray,(5,5),7)
    
    # compute R value
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    
    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
    
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    
    R = np.divide(det, trace)
    
    return R

# question 1-b
def perform_non_maximal_suppression(img, radius):
    img_copy = img.copy()
    #print(img_copy.shape)
    num_rows, num_columns = img_copy.shape
    
    for i in range(0, num_rows):
        for j in range(0, num_columns):
            left = i - radius
            right = i + radius
            top = j - radius
            bottom = j + radius
            
            # update the patch index; if patch index out of index (boundary) of
            # image, then we need to update it
            if left < 0:
                left = 0
            if right > (num_rows - 1):
                right = num_rows - 1
            if top < 0:
                top = 0
            if bottom > (num_columns - 1):
                bottom = num_columns - 1
                
            # find the local maxima and update intensity value for each pixel in
            # image
            cur_intensity = img_copy[i][j]
            patch = img_copy[left : right + 1, top : bottom + 1]
            max_intensity = np.amax(patch)
            if cur_intensity != max_intensity:
                img_copy[i][j] = 0
                
    return img_copy
    
# question 1-c
def do_si_kp_detection():
    gray = cv2.imread('./synthetic.png', 0)
    synthetic_img_copy = cv2.imread('./synthetic.png').copy()
    sigma_list = [3, 5, 9, 12, 25, 30, 35]
    img_result = []
    rows, columns = gray.shape
    result = np.zeros((rows, columns))
    
    for sigma in sigma_list:
        #gau_img = cv2.GaussianBlur(gray, (5, 5), 5)
        lap_img = gaussian_laplace(gray, sigma)
        nms_img = perform_non_maximal_suppression(lap_img, 50)
        lap_img[lap_img < 0.035] = 0
        lap_img[lap_img > 10] = 0
        img_result.append(nms_img)
    
    # find maxima among 5 images that have gaussian lapacian applied
    for i in range(0, rows):
        for j in range(0, columns):
            max = 0
            for processed_img in img_result:
                temp = processed_img[i][j]
                if temp > max:
                    max = temp
            result[i][j] = max
    
    # add circle on origianl image to circle out the interest points
    for s in range(0, result.shape[0]):
        for t in range(0, result.shape[1]):
            if not (result[s][t] == 0):
                cv2.circle(synthetic_img_copy, (t,s), 10, (255, 0, 0))
                
    cv2.imwrite("./si_kp_detection.jpg", synthetic_img_copy)
    
    #return result
    
# question 1-d
def fast_feature_descriptor():
    building = cv2.imread('./building.jpg').copy()
    synthetic = cv2.imread('./synthetic.png').copy()
    building_gray = cv2.imread('./building.jpg', 0)
    synthetic_gray = cv2.imread('./synthetic.png', 0)
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    
    # find and draw the keypoints
    kp_for_building = fast.detect(building_gray, None)
    kp_for_synthetic = fast.detect(synthetic_gray, None)
    cv2.drawKeypoints(building, kp_for_building, building, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(synthetic, kp_for_synthetic, synthetic, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite('fast_building.jpg', building)
    cv2.imwrite('fast_sythetic.jpg', synthetic)
    
# question 2-a
def apply_sift():
    book = cv2.imread('./book.jpeg').copy()
    findBook = cv2.imread('./findBook.png').copy()
    
    book_gray = cv2.imread('./book.jpeg', 0)
    findBook_gray = cv2.imread('./findBook.png', 0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp_for_book, des_for_book = sift.detectAndCompute(book_gray, None)
    kp_for_findBook, des_for_findBook = sift.detectAndCompute(findBook_gray, None)
    
    #print des_for_book
    #print des_for_findBook
    
    cv2.drawKeypoints(book, kp_for_book, book, color=(255,0,0),  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(findBook, kp_for_findBook, findBook, color=(255,0,0),  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite('./book_sift.jpg', book)
    cv2.imwrite('./findBook_sift.jpg', findBook)
    
    return [(kp_for_book, des_for_book), (kp_for_findBook, des_for_findBook)]
    
# question 2-b
def match_algorithm(kp_and_des_list):
    img1 = cv2.imread('./book.jpeg').copy()
    img2 = cv2.imread('./findBook.png').copy()
    book_kp, book_des = kp_and_des_list[0]
    findBook_kp, findBook_des = kp_and_des_list[1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_of_good_matches = []
    matches_found = []
    
    for index in range(len(thresholds)):
        matches_found.append([])
    
    for i in range(0, book_des.shape[0]):
        distance_values_list = []
        kp_list = []
        
        for j in range(0, findBook_des.shape[0]):
            des1 = book_des[i, : ]
            des2 = findBook_des[j, : ]
            kp1 = book_kp[i]
            kp2 = findBook_kp[j]
            
            dist = distance.euclidean(des1, des2)
            
            distance_values_list.append(dist)
            kp_list.append((i, j))
        
        # get the minimum distance, whcih is the closest descriptor
        min_value = min(distance_values_list)
        min_index = distance_values_list.index(min_value)
        kp_tuple = kp_list[min_index]
        distance_values_list.pop(min_index)
        kp_list.pop(min_index)
        
        # get the second minimum distance, whcih is the second closest descriptor
        second_min = min(distance_values_list)
        second_min_index = distance_values_list.index(second_min)
        second_kp_tuple = kp_list[second_min_index]
        
        ratio = min_value / float(second_min)
        for l in range(len(thresholds)):
            if ratio < thresholds[l]:
                match = cv2.DMatch(kp_tuple[0], kp_tuple[1], min_value)
                matches_found[l].append(match)
    
    for match in matches_found:
        num_of_good_matches.append(len(match))
        
    plt.plot(thresholds, num_of_good_matches, color="green")
    plt.title("Number of Matches vs. Thresholds")
    plt.xlabel('Threshold')
    plt.ylabel('Number of Matches')
    plt.show()
        
    #match_img = cv2.drawMatches(img1, book_kp, img2, findBook_kp, good_matches, None)
    #cv2.imwrite('./match_book.jpg', match_img)
    
# question 2-c
def solve_affine_transformation():
    # get keypoints
    kp_and_des_list = apply_sift()
    book_kp, book_des = kp_and_des_list[0]
    findBook_kp, findBook_des = kp_and_des_list[1]
    
    # get matches
    matches_found = []
    
    for i in range(0, book_des.shape[0]):
        distance_values_list = []
        kp_list = []
        
        for j in range(0, findBook_des.shape[0]):
            des1 = book_des[i, : ]
            des2 = findBook_des[j, : ]
            kp1 = book_kp[i]
            kp2 = findBook_kp[j]
            
            dist = distance.euclidean(des1, des2)
            
            distance_values_list.append(dist)
            kp_list.append((i, j))
        
        # get the minimum distance, whcih is the closest descriptor
        min_value = min(distance_values_list)
        min_index = distance_values_list.index(min_value)
        kp_tuple = kp_list[min_index]
        distance_values_list.pop(min_index)
        kp_list.pop(min_index)
        
        # get the second minimum distance, whcih is the second closest descriptor
        second_min = min(distance_values_list)
        second_min_index = distance_values_list.index(second_min)
        second_kp_tuple = kp_list[second_min_index]
        
        ratio = min_value / float(second_min)
        if ratio < 0.6:
            match = cv2.DMatch(kp_tuple[0], kp_tuple[1], min_value)
            matches_found.append(match)
            
    sorted_matches = sorted(matches_found, key=lambda match: match.distance)
            
    k_values = [3, 5, 7, 10]
    affines = []
    
    for index in range(len(k_values)):
        k = k_values[index]
        p = np.zeros((2*k, 6))
        p_transform = np.zeros((2*k, 1))
        
        next_index = 0
        for l in range(k):
            p_index_x = book_kp[sorted_matches[l].queryIdx].pt[0]
            p_index_y = book_kp[sorted_matches[l].queryIdx].pt[1]
            p[next_index, 0] = p_index_x
            p[next_index, 1] = p_index_y
            p[next_index, 4] = 1
            p[next_index + 1, 2] = p_index_x
            p[next_index + 1, 3] = p_index_y
            p[next_index + 1, 5] = 1
            
            p_transform_x = findBook_kp[sorted_matches[l].trainIdx].pt[0]
            p_transform_y = findBook_kp[sorted_matches[l].trainIdx].pt[1]
            p_transform[next_index, 0] = p_transform_x
            p_transform[next_index + 1, 0] = p_transform_y
            
            next_index = next_index + 2
            
        ptp = np.dot(p.T, p)
        inverse = np.linalg.pinv(ptp)
        temp = np.dot(inverse.T, p.T)
        affine = np.dot(temp, p_transform)
        affines.append(affine)
        print('k = {0}, {1}'.format(k, affine))
    return affines
    
# question 2-d
def visualize(affines):
    img1 = cv2.imread('./book.jpeg').copy()
    img2 = cv2.imread('./findBook.png').copy()
    
    # take affine when k equals to 10
    #print(affines)
    affine = affines[3]
    top = 0
    left = 0
    right = img1.shape[1] - 1
    bottom = img1.shape[0] - 1
    
    top_left = np.array([
        [left, top, 0, 0, 1, 0],
        [0, 0, left, top, 0, 1]
    ])
    top_right = np.array([
        [right, top, 0, 0, 1, 0],
        [0, 0, right, top, 0, 1]
    ])
    bottom_left = np.array([
        [left, bottom, 0, 0, 1, 0],
        [0, 0, left, bottom, 0, 1]
    ])
    bottom_right = np.array([
        [right, bottom, 0, 0, 1, 0],
        [0, 0, right, bottom, 0, 1]
    ])
    
    trans_top_left = np.dot(top_left, affine).reshape((1,2))
    trans_top_right = np.dot(top_right, affine).reshape((1,2))
    trans_bottom_left = np.dot(bottom_left, affine).reshape((1,2))
    trans_bottom_right = np.dot(bottom_right, affine).reshape((1,2))
    
    corner_points = np.array([trans_top_left, trans_bottom_left, trans_bottom_right, trans_top_right,], np.int32)
    result = cv2.polylines(img2, [corner_points], True, (255, 0, 0), 3)
    
    cv2.imwrite('./2d.jpg', result)
    return corner_points
    
# helper function for 2-e
def isColorSame(c1, c2):
    for i in range(3):
        if c1[i] != c2[i]:
            return False
    return True
    
# question 2-e
def match_with_color():
    search = cv2.imread('./colourSearch.png').copy()
    temp = cv2.imread('./colourTemplate.png').copy()
    search_gray = cv2.imread('./colourSearch.png', 0)
    temp_gray = cv2.imread('./colourTemplate.png', 0)
    
    # do sift on grayscale images to get keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_for_temp, des_for_temp = sift.detectAndCompute(temp_gray, None)
    kp_for_search, des_for_search = sift.detectAndCompute(search_gray, None)
    
    # find matches
    matches_found = []
    
    for i in range(0, des_for_temp.shape[0]):
        distance_values_list = []
        kp_list = []
        
        for j in range(0, des_for_search.shape[0]):
            des1 = des_for_temp[i, : ]
            des2 = des_for_search[j, : ]
            kp1 = kp_for_temp[i]
            kp2 = kp_for_search[j]
            
            dist = distance.euclidean(des1, des2)
            
            distance_values_list.append(dist)
            kp_list.append((i, j))
        
        # get the minimum distance, whcih is the closest descriptor
        min_value = min(distance_values_list)
        min_index = distance_values_list.index(min_value)
        kp_tuple = kp_list[min_index]
        distance_values_list.pop(min_index)
        kp_list.pop(min_index)
        
        # get the second minimum distance, whcih is the second closest descriptor
        second_min = min(distance_values_list)
        second_min_index = distance_values_list.index(second_min)
        second_kp_tuple = kp_list[second_min_index]
        
        ratio = min_value / float(second_min)
        if ratio < 0.6:
            temp_x = int(kp_for_temp[kp_tuple[0]].pt[0])
            temp_y = int(kp_for_temp[kp_tuple[0]].pt[1])
            search_x = int(kp_for_search[kp_tuple[1]].pt[0])
            search_y = int(kp_for_search[kp_tuple[1]].pt[1])
            
            temp_color = temp[temp_x][temp_y]
            search_color = search[search_x][search_y]
            
            if isColorSame(temp_color, search_color):
                match = cv2.DMatch(kp_tuple[0], kp_tuple[1], min_value)
                matches_found.append(match)
                
    sorted_matches = sorted(matches_found, key=lambda match: match.distance)
    #print(sorted_matches)
                
    # find affine
    k = 4
    p = np.zeros((2*k, 6))
    p_transform = np.zeros((2*k, 1))
    
    next_index = 0
    for l in range(k):
        p_index_x = kp_for_temp[sorted_matches[l].queryIdx].pt[0]
        p_index_y = kp_for_temp[sorted_matches[l].queryIdx].pt[1]
        p[next_index, 0] = p_index_x
        p[next_index, 1] = p_index_y
        p[next_index, 4] = 1
        p[next_index + 1, 2] = p_index_x
        p[next_index + 1, 3] = p_index_y
        p[next_index + 1, 5] = 1
        
        p_transform_x = kp_for_search[sorted_matches[l].trainIdx].pt[0]
        p_transform_y = kp_for_search[sorted_matches[l].trainIdx].pt[1]
        p_transform[next_index, 0] = p_transform_x
        p_transform[next_index + 1, 0] = p_transform_y
        
        next_index = next_index + 2
        
    ptp = np.dot(p.T, p)
    inverse = np.linalg.pinv(ptp)
    temp = np.dot(inverse.T, p.T)
    affine = np.dot(temp, p_transform)
    
    #print(affine)
    
    # do transformation
    top = 0
    left = 0
    right = temp_gray.shape[1] - 1
    bottom = temp_gray.shape[0] - 1
    
    top_left = np.array([
        [left, top, 0, 0, 1, 0],
        [0, 0, left, top, 0, 1]
    ])
    top_right = np.array([
        [right, top, 0, 0, 1, 0],
        [0, 0, right, top, 0, 1]
    ])
    bottom_left = np.array([
        [left, bottom, 0, 0, 1, 0],
        [0, 0, left, bottom, 0, 1]
    ])
    bottom_right = np.array([
        [right, bottom, 0, 0, 1, 0],
        [0, 0, right, bottom, 0, 1]
    ])
    
    trans_top_left = np.dot(top_left, affine).reshape((1,2))
    trans_top_right = np.dot(top_right, affine).reshape((1,2))
    trans_bottom_left = np.dot(bottom_left, affine).reshape((1,2))
    trans_bottom_right = np.dot(bottom_right, affine).reshape((1,2))
    
    corner_points = np.array([trans_top_left, trans_bottom_left, trans_bottom_right, trans_top_right,], np.int32)
    #print(corner_points)
    result = cv2.polylines(search, [corner_points], True, (0, 255, 255), 3)
    
    cv2.imwrite('./2e.jpg', result)
    #return corner_points    

# question 3-a
def get_plot_3a():
    s_values = []
    k_values = []
    P = 0.99
    p = 0.7
    
    for k_value in range(1, 21):
        s = log(1 - P) / log(1 - (p**(k_value)))
        s_values.append(s)
        k_values.append(k_value)
        
    plt.plot(k_values, s_values)
    plt.title('S vs. k')
    plt.xlabel('k')
    plt.ylabel('S')
    plt.show()
        
# question 3-b
def get_plot_3b():
    s_values = []
    p_values = []
    P = 0.99
    k_value = 5
    
    for p in range(1, 6):
        p = p / float(10)
        s = log(1 - P) / log(1 - (p**(k_value)))
        s_values.append(s)
        p_values.append(p)
        
    plt.plot(p_values, s_values)
    plt.title('S vs. p')
    plt.xlabel('p')
    plt.ylabel('S')
    plt.show()

# main function is defined here
if __name__ == "__main__":
    # question 1-a; harris method
    building_img = cv2.imread('./building.jpg')
    harris_R = compute_harris_method(building_img)
    cv2.normalize(harris_R, harris_R, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("./harris.jpg", harris_R)
    
    # question 1-a; brown (harmonic mean) method
    brown_R = compute_brown_method(building_img)
    cv2.normalize(brown_R, brown_R, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("./brown.jpg", brown_R)
    
    # question 1-b
    radius = [1, 3, 5, 9]
    brown_building = compute_brown_method(building_img)
    for r in radius:
        nms_result = perform_non_maximal_suppression(brown_building, r)
        file_name = './brown_nms{0}.jpg'.format(r)
        cv2.imwrite(file_name, nms_result)
    
    # question 1-c
    do_si_kp_detection()
    
    # question 1-d
    fast_feature_descriptor()
    
    # question 2-a
    kp_and_des_list = apply_sift()
    
    # question 2-b
    match_algorithm(kp_and_des_list)
    
    # question 2-c
    affines = solve_affine_transformation()
    
    # question 2-d
    corner_points = visualize(affines)
    
    # question 2-e
    match_with_color()
    
    # question 3-a
    get_plot_3a()
    
    # question 3-b
    get_plot_3b()
    
