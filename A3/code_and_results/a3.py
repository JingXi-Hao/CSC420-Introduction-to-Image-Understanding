import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
import math

# question 1
def estimate_width_height_using_homography():
    # read the images first
    door = cv.imread('./door.jpg').copy()
    # door_gray = cv.imread('./door.jpg', 0)
    
    # find the coordinates
    cv.circle(door, (171, 219), 6, (255, 0, 0)) # top left corner
    cv.circle(door, (170, 274), 6, (255, 0, 0)) # bottom left corner
    cv.circle(door, (205, 273), 6, (255, 0, 0)) # bottom right corner
    cv.circle(door, (205, 214), 6, (255, 0, 0)) # top right corner
    # cv.imwrite('./door_dot.jpg', door)
    
    # transformed points
    # y - value
    top = 0
    bottom = 279
    # x - value
    left = 0
    right = 216
    
    # define the matrix A --- 215.6 = 816 pixels
    # 279.4 = 1056 pixels
    A = np.array([
        [171, 219, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 171, 219, 1, 0, 0, 0],
        [170, 274, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 170, 274, 1, -47430, -76446, -279],
        [205, 273, 1, 0, 0, 0, -44280, -58968, -216],
        [0, 0, 0, 205, 273, 1, -57195, -76167, -279],
        [205, 214, 1, 0, 0, 0, -44280, -46224, -216],
        [0, 0, 0, 205, 214, 1, 0, 0, 0]
    ])
    
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(A.T, A))
    # print(np.dot(A.T, A))
    # print(eigen_values)
    # print(eigen_vectors)
    
    min_eigen_value = np.amin(eigen_values)
    # print(min_eigen_value)
    
    min_index = eigen_values.tolist().index(min_eigen_value)
    #print(min_index)
    
    min_eigen_vector = eigen_vectors[ : , min_index]
    # print(min_eigen_vector)
    
    # find h values and reshape it as 3 by 3 matrix
    h = min_eigen_vector.reshape((3, 3))
    # print(min_eigen_vector)
    # print(h)
    
    # coordinates for door
    cv.circle(door, (118, 89), 6, (0, 255, 0)) # top left corner
    cv.circle(door, (118, 553), 6, (0, 255, 0)) # bottom left corner
    cv.circle(door, (257, 631), 6, (0, 255, 0)) # bottom right corner
    cv.circle(door, (257, 31), 6, (0, 255, 0)) # top right corner
    cv.imwrite('./door_with_circle.jpg', door)
    
    # find transformed coordinates
    # top left point
    top_left = np.array([
        [118, 89, 1]
    ]).reshape((3, 1))
    #print(top_left)
    # bottom left point
    bottom_left = np.array([
        [118, 553, 1]
    ]).reshape((3, 1))
    # bottom right point
    bottom_right = np.array([
        [257, 631, 1]
    ]).reshape((3, 1))
    # top right point
    top_right = np.array([
        [257, 31, 1]
    ]).reshape((3, 1))
    
    # test here
    # test_top_left = np.dot(h, np.array([171, 219, 1]).reshape((3, 1)))
    # print(test_top_left)
    
    # transformed top left point
    transformed_top_left = np.dot(h, top_left)
    transformed_top_left = ((transformed_top_left / transformed_top_left[2, 0])[0 : 2, 0]).reshape((2, 1))
    #print(transformed_top_left)
    
    # transformed bottom left point
    transformed_bottom_left = np.dot(h, bottom_left)
    transformed_bottom_left = ((transformed_bottom_left / transformed_bottom_left[2, 0])[0 : 2, 0]).reshape((2, 1))
    #print(transformed_bottom_left)
    
    # transformed bottom right point
    transformed_bottom_right = np.dot(h, bottom_right)
    transformed_bottom_right = ((transformed_bottom_right / transformed_bottom_right[2, 0])[0 : 2, 0]).reshape((2, 1))
    #print(transformed_bottom_right)
    
    # transformed top right point
    transformed_top_right = np.dot(h, top_right)
    transformed_top_right = ((transformed_top_right / transformed_top_right[2, 0])[0 : 2, 0]).reshape((2, 1))
    #print(transformed_top_right)
    
    #plt.scatter([-469, -298, 447, 520], [-894, 1591, 1555, -846])
    #plt.show()
    
    width = np.linalg.norm(transformed_top_left - transformed_top_right)
    #width2 = np.linalg.norm(transformed_bottom_left - transformed_bottom_right)
    print('The width is {} mm'.format(width))
    #print(width2)
    
    height = np.linalg.norm(transformed_top_left - transformed_bottom_left)
    #height2 = np.linalg.norm(transformed_top_right - transformed_bottom_right)
    print('The height is {} mm'.format(height))
    #print(height2)
    
# helper function for question 2a
def get_good_matches(matches, ratio):
    good = []
    
    # Apply ratio test
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    return good

# question 2a
def find_matches():
    # read the images first
    bookCover = cv.imread("./bookCover.jpg").copy()
    im1 = cv.imread("./im1.jpg").copy()
    im2 = cv.imread("./im2.jpg").copy()
    im3 = cv.imread("./im3.jpg").copy()
    
    # get some gray images
    bookCover_gray = cv.imread("./bookCover.jpg", 0)
    im1_gray = cv.imread("./im1.jpg", 0)
    im2_gray = cv.imread("./im2.jpg", 0)
    im3_gray = cv.imread("./im3.jpg", 0)
    
    # create the sift detector
    sift = cv.xfeatures2d.SIFT_create()
    
    # for img1
    kp_book_cover, des_book_cover = sift.detectAndCompute(bookCover_gray, None)
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    
    # for img2
    kp2, des2 = sift.detectAndCompute(im2_gray, None)
    
    # for img3
    kp3, des3 = sift.detectAndCompute(im3_gray, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches1 = bf.knnMatch(des_book_cover, des1, k=2)
    
    # get good matches
    good_matches1 = get_good_matches(matches1, 0.6)
    #print(len(good_matches1))
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    result1 = cv.drawMatchesKnn(bookCover_gray, kp_book_cover, im1_gray, kp1, good_matches1, None, flags=2)
    
    # write the result in file
    cv.imwrite('./match_im1.jpg', result1)
    
    # matches for img2 and draw it out
    matches2 = bf.knnMatch(des_book_cover, des2, k=2)
    good_matches2 = get_good_matches(matches2, 0.6)
    #print(len(good_matches2))
    result2 = cv.drawMatchesKnn(bookCover_gray, kp_book_cover, im2_gray, kp2, good_matches2, None, flags=2)
    cv.imwrite('./match_im2.jpg', result2)
    
    # matches for img3 and draw it out
    matches3 = bf.knnMatch(des_book_cover, des3, k=2)
    good_matches3 = get_good_matches(matches3, 0.6)
    #print(len(good_matches3))
    result3 = cv.drawMatchesKnn(bookCover_gray, kp_book_cover, im3_gray, kp3, good_matches3, None, flags=2)
    cv.imwrite('./match_im3.jpg', result3)
    
# helper function for question 2c
def get_src_and_dst_coordinates(kp_book_cover, kp_img, good_matches):
    src = []
    dst = []
    
    for i in range(len(good_matches)):
        match = good_matches[i][0]
        #print(match)
        src_x = kp_book_cover[match.queryIdx].pt[0]
        src_y = kp_book_cover[match.queryIdx].pt[1]
        dst_x = kp_img[match.trainIdx].pt[0]
        dst_y = kp_img[match.trainIdx].pt[1]
        
        src.append((src_x, src_y))
        dst.append((dst_x, dst_y))
        
    return (src, dst)
    
# question 2c
def find_affine_transformation():
    # read the images first
    bookCover = cv.imread("./bookCover.jpg").copy()
    im1 = cv.imread("./im1.jpg").copy()
    im2 = cv.imread("./im2.jpg").copy()
    im3 = cv.imread("./im3.jpg").copy()
    
    # get some gray images
    bookCover_gray = cv.imread("./bookCover.jpg", 0)
    im1_gray = cv.imread("./im1.jpg", 0)
    im2_gray = cv.imread("./im2.jpg", 0)
    im3_gray = cv.imread("./im3.jpg", 0)
    
    # create the sift detector
    sift = cv.xfeatures2d.SIFT_create()
    
    # for img1 and bookCover
    kp_book_cover, des_book_cover = sift.detectAndCompute(bookCover_gray, None)
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    
    # for img2
    kp2, des2 = sift.detectAndCompute(im2_gray, None)
    
    # for img3
    kp3, des3 = sift.detectAndCompute(im3_gray, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    
    # matches for img2 and draw it out
    matches1 = bf.knnMatch(des_book_cover, des1, k=2)
    good_matches1 = get_good_matches(matches1, 0.6)
    src1, dst1 = get_src_and_dst_coordinates(kp_book_cover, kp1, good_matches1)
    affine1, inliers1 = cv.estimateAffine2D(np.array(src1), np.array(dst1), method=cv.RANSAC)
    warp_im1 = cv.warpAffine(bookCover, affine1, (im1.shape[1], im1.shape[0]))
    black_index = np.sum(warp_im1, axis = 2) == 0
    warp_im1[black_index] = im1[black_index]
    cv.imwrite('./warp_affine_img1.jpg', warp_im1)
    
    # matches for img2 and draw it out
    matches2 = bf.knnMatch(des_book_cover, des2, k=2)
    good_matches2 = get_good_matches(matches2, 0.6)
    src2, dst2 = get_src_and_dst_coordinates(kp_book_cover, kp2, good_matches2)
    affine2, inliers2 = cv.estimateAffine2D(np.array(src2), np.array(dst2), method=cv.RANSAC)
    warp_im2 = cv.warpAffine(bookCover, affine2, (im2.shape[1], im2.shape[0]))
    black_index = np.sum(warp_im2, axis = 2) == 0
    warp_im2[black_index] = im2[black_index]
    cv.imwrite('./warp_affine_img2.jpg', warp_im2)
    
    # matches for img3 and draw it out
    matches3 = bf.knnMatch(des_book_cover, des3, k=2)
    good_matches3 = get_good_matches(matches3, 0.6)
    src3, dst3 = get_src_and_dst_coordinates(kp_book_cover, kp3, good_matches3)
    affine3, inliers3 = cv.estimateAffine2D(np.array(src3), np.array(dst3), method=cv.RANSAC)
    warp_im3 = cv.warpAffine(bookCover, affine3, (im3.shape[1], im3.shape[0]))
    black_index = np.sum(warp_im3, axis = 2) == 0
    warp_im3[black_index] = im3[black_index]
    cv.imwrite('./warp_affine_img3.jpg', warp_im3)
    
# question 2d
def find_homography():
    # read the images first
    bookCover = cv.imread("./bookCover.jpg").copy()
    im1 = cv.imread("./im1.jpg").copy()
    im2 = cv.imread("./im2.jpg").copy()
    im3 = cv.imread("./im3.jpg").copy()
    
    # get some gray images
    bookCover_gray = cv.imread("./bookCover.jpg", 0)
    im1_gray = cv.imread("./im1.jpg", 0)
    im2_gray = cv.imread("./im2.jpg", 0)
    im3_gray = cv.imread("./im3.jpg", 0)
    
    # create the sift detector
    sift = cv.xfeatures2d.SIFT_create()
    
    # for img1 and bookCover
    kp_book_cover, des_book_cover = sift.detectAndCompute(bookCover_gray, None)
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    
    # for img2
    kp2, des2 = sift.detectAndCompute(im2_gray, None)
    
    # for img3
    kp3, des3 = sift.detectAndCompute(im3_gray, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    
    # matches for img2 and draw it out
    matches1 = bf.knnMatch(des_book_cover, des1, k=2)
    good_matches1 = get_good_matches(matches1, 0.6)
    src1, dst1 = get_src_and_dst_coordinates(kp_book_cover, kp1, good_matches1)
    homography1, mask1 = cv.findHomography(np.array(src1), np.array(dst1), cv.RANSAC, 5.0)
    warp_im1 = cv.warpPerspective(bookCover, homography1, (im1.shape[1], im1.shape[0]))
    black_index = np.sum(warp_im1, axis = 2) == 0
    warp_im1[black_index] = im1[black_index]
    cv.imwrite('./warp_homo_img1.jpg', warp_im1)
    
    # matches for img2 and draw it out
    matches2 = bf.knnMatch(des_book_cover, des2, k=2)
    good_matches2 = get_good_matches(matches2, 0.6)
    src2, dst2 = get_src_and_dst_coordinates(kp_book_cover, kp2, good_matches2)
    homography2, mask2 = cv.findHomography(np.array(src2), np.array(dst2), cv.RANSAC, 5.0)
    warp_im2 = cv.warpPerspective(bookCover, homography2, (im2.shape[1], im2.shape[0]))
    black_index = np.sum(warp_im2, axis = 2) == 0
    warp_im2[black_index] = im2[black_index]
    cv.imwrite('./warp_homo_img2.jpg', warp_im2)
    
    # matches for img3 and draw it out
    matches3 = bf.knnMatch(des_book_cover, des3, k=2)
    good_matches3 = get_good_matches(matches3, 0.6)
    src3, dst3 = get_src_and_dst_coordinates(kp_book_cover, kp3, good_matches3)
    homography3, mask3 = cv.findHomography(np.array(src3), np.array(dst3), cv.RANSAC, 5.0)
    warp_im3 = cv.warpPerspective(bookCover, homography3, (im3.shape[1], im3.shape[0]))
    black_index = np.sum(warp_im3, axis = 2) == 0
    warp_im3[black_index] = im3[black_index]
    cv.imwrite('./warp_homo_img3.jpg', warp_im3)
    
# question 2e
def map_another_book_cover():
    # read the images first
    bookCover = cv.imread("./anotherBookCover.jpg").copy()
    im1 = cv.imread("./im1.jpg").copy()
    im2 = cv.imread("./im2.jpg").copy()
    im3 = cv.imread("./im3.jpg").copy()
    
    # get some gray images
    bookCover_gray = cv.imread("./anotherBookCover.jpg", 0)
    im1_gray = cv.imread("./im1.jpg", 0)
    im2_gray = cv.imread("./im2.jpg", 0)
    im3_gray = cv.imread("./im3.jpg", 0)
    
    # create the sift detector
    sift = cv.xfeatures2d.SIFT_create()
    
    # for img1 and bookCover
    kp_book_cover, des_book_cover = sift.detectAndCompute(bookCover_gray, None)
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    
    # for img2
    kp2, des2 = sift.detectAndCompute(im2_gray, None)
    
    # for img3
    kp3, des3 = sift.detectAndCompute(im3_gray, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    
    # matches for img2 and draw it out
    matches1 = bf.knnMatch(des_book_cover, des1, k=2)
    good_matches1 = get_good_matches(matches1, 0.8)
    src1, dst1 = get_src_and_dst_coordinates(kp_book_cover, kp1, good_matches1)
    homography1, mask1 = cv.findHomography(np.array(src1), np.array(dst1), cv.RANSAC, 5.0)
    warp_im1 = cv.warpPerspective(bookCover, homography1, (im1.shape[1], im1.shape[0]))
    black_index = np.sum(warp_im1, axis = 2) == 0
    warp_im1[black_index] = im1[black_index]
    cv.imwrite('./another_warp_homo_img1.jpg', warp_im1)
    
    # matches for img2 and draw it out
    matches2 = bf.knnMatch(des_book_cover, des2, k=2)
    good_matches2 = get_good_matches(matches2, 0.8)
    src2, dst2 = get_src_and_dst_coordinates(kp_book_cover, kp2, good_matches2)
    homography2, mask2 = cv.findHomography(np.array(src2), np.array(dst2), cv.RANSAC, 5.0)
    warp_im2 = cv.warpPerspective(bookCover, homography2, (im2.shape[1], im2.shape[0]))
    black_index = np.sum(warp_im2, axis = 2) == 0
    warp_im2[black_index] = im2[black_index]
    cv.imwrite('./another_warp_homo_img2.jpg', warp_im2)
    
    # matches for img3 and draw it out
    matches3 = bf.knnMatch(des_book_cover, des3, k=2)
    good_matches3 = get_good_matches(matches3, 0.8)
    src3, dst3 = get_src_and_dst_coordinates(kp_book_cover, kp3, good_matches3)
    homography3, mask3 = cv.findHomography(np.array(src3), np.array(dst3), cv.RANSAC, 5.0)
    warp_im3 = cv.warpPerspective(bookCover, homography3, (im3.shape[1], im3.shape[0]))
    black_index = np.sum(warp_im3, axis = 2) == 0
    warp_im3[black_index] = im3[black_index]
    cv.imwrite('./another_warp_homo_img3.jpg', warp_im3)
    
# question 3
def find_matrix_for_camera():
    result_matrix = np.zeros((3, 3)) # result matrix K
    bottle = cv.imread('./bottle.jpg').copy()
    bottle_gray = cv.imread('./bottle.jpg', 0)
    height, width, color = bottle.shape
    #print(height)
    #print(width)
    
    # get height on image in pixel
    height_on_image_half = height / 2.0
    
    # find the principal point, p, coordinates, assume the principal point is the
    # center of the image in the unit of pixel
    p_x = round(width / 2.0)
    p_y = round(height / 2.0)
    
    # measure and record the actual height of the bottle in mm
    # (15 + 3.5)cm
    actual_height_half = (18.5 * 10) / 2.0
    #print(actual_height)
    
    # measure the distance from the object to the camera and convert into mm
    total_width = 17 * 10
    #print(total_width)
    
    # then, based on the information above, we are able to compute the focal
    # length
    f = (height_on_image_half / float(actual_height_half)) * total_width
    
    # replace the value in result matrix K to get K
    result_matrix[0][0] = f
    result_matrix[1][1] = f
    result_matrix[0][2] = p_x
    result_matrix[1][2] = p_y
    result_matrix[2][2] = 1
    
    print('The estimate of the internal parameter matrix K is')
    print(result_matrix)
    
if __name__ == "__main__":
    # question 1
    estimate_width_height_using_homography()
    
    # quesrion 2a - by visualizing, we have 3 outliers for im1, we have 
    find_matches()
    
    # question 2c
    find_affine_transformation()
    
    # question 2d
    find_homography()
    
    # question 2e
    map_another_book_cover()
    
    # question 3
    find_matrix_for_camera()
    
