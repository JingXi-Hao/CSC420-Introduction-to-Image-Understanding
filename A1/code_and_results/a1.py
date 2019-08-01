import numpy as np
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter, gaussian_laplace

# question 1
# this is a helper function for question 1, which adds zero padding for image
# for full and same mode
def doZeroPaddingForImage(img, imgRows, imgColumns, rowPadding, columnPadding, isColorImage):
    if not isColorImage:
        # add zero padding for grayscale image
        paddedImg = np.pad(img, rowPadding, "constant")
    else:
        # add zero padding for color image
        paddedImgRows = imgRows + (rowPadding * 2)
        paddedImgColumns = imgColumns + (columnPadding * 2)
        paddedImg = np.zeros((paddedImgRows, paddedImgColumns, 3))
        
        # reassign the values in the paddedImage
        for t in range(0, imgRows):
            for s in range(0, imgColumns):
                newT = t + rowPadding
                newS = s + columnPadding
                paddedImg[newT][newS] = img[t][s]
        
    return paddedImg

# this is a helper function for question 1, which computes sum for single pixel
def computeSum(filter, partialImg):
    sum = 0
    for p in range(0, filter.shape[0]):
        for q in range(0, filter.shape[1]):
           sum = sum + filter[p][q] * partialImg[p][q]
    return sum

# this is a helper function for question 1, which generate the result image
def generateResultImage(rows, columns, filterRows, filterColumns, paddedImg, filter):
    resultImg = []
    
    for k in range(0, rows):
        newRow = []
        for l in range(0, columns):
            sum = computeSum(filter, paddedImg[k : k+filterRows, l : l+filterColumns])
            newRow.append(sum)
        resultImg.append(newRow)
            
    result = np.array(resultImg)
    return result

# this function is for question 1, which computes the correlation between an 
# input image and a given correlation filter
def computeCorrelation(I, f, mode):
    # drop the redundant information gotten from imread
    img = np.array(I.tolist())
    
    # get the shape of input grayscale or color img and 2D filter, assume filter
    # is a square, so rowK equals to columnK
    filterRows, filterColumns = f.shape
    rowK = filterRows // 2
    columnK = filterColumns // 2
    
    if len(img.shape) == 3:
        # img is a color image
        isColorImage = True
        imgRows = img.shape[0]
        imgColumns = img.shape[1]
    else:
        # img is a grayscale image
        isColorImage = False
        imgRows, imgColumns = img.shape
    
    if mode == "same":
        # do zero padding for the img to generate a padded img
        # compute dimension for padded img
        paddedImg = doZeroPaddingForImage(img, imgRows, imgColumns, rowK, columnK, isColorImage)
        
        # compute the correlation
        result = generateResultImage(imgRows, imgColumns, filterRows, filterColumns, paddedImg, f)
    
    elif mode == "full":
        # do zero padding for the img to generate a padded img
        # compute dimension for padded img
        rowPadding = filterRows - 1
        columnPadding = filterColumns - 1
        paddedImg = doZeroPaddingForImage(img, imgRows, imgColumns, rowPadding, columnPadding, isColorImage)
        
        # compute the correlation
        rows = imgRows + (rowPadding * 2) - (rowK * 2)
        columns = imgColumns + (columnPadding * 2) - (columnK * 2) 
        result = generateResultImage(rows, columns, filterRows, filterColumns, paddedImg, f)
        
    elif mode == "valid":
        # compute the correlation
        rows = imgRows - (rowK * 2)
        columns = imgColumns - (columnK * 2) 
        result = generateResultImage(rows, columns, filterRows, filterColumns, img, f)
    
    return result
    
    
# question 2
def doConvolutionUsingCorrelation(img, varX, varY, mode):
    # define two 1D gaussian filters each with variance equals to 3 and 5
    horizontalFilter = cv.getGaussianKernel(29, 3)
    verticalFilter = cv.getGaussianKernel(29, 5)
    
    # apply dot product on these two 1D filters and we get a 2D gaussian filter
    gaussianFilter = np.dot(verticalFilter, np.transpose(horizontalFilter))
    
    # flip the filter and pass flippedFilter as parameter into 
    # computeCorrelation
    flippedFilter = np.flipud(np.fliplr(gaussianFilter))
    
    result = computeCorrelation(img, flippedFilter, mode)
    cv.imwrite("./result2.jpg", result)
    

# question 7
def applyGaussionFilter():
    portrait = cv.imread("./portrait.jpg")
    portraitGray = cv.cvtColor(portrait, cv.COLOR_BGR2GRAY)
    
    # apply derivative of Gaussion filter
    result7_derivative = gaussian_filter(portraitGray, 3, order=[1, 1])
    cv.imwrite("./result7_derivative.jpg", result7_derivative)
    
    # apply Laplacian of Gaussian filter
    result7_lap = gaussian_laplace(portraitGray, sigma=3)
    cv.imwrite("./result7_lap.jpg", result7_lap)

    
# question 8
def findWaldo():
    # save the original images and save the graysacle ones as well
    wheresWaldo = cv.imread("./whereswaldo.jpg")
    wheresWaldoOriginal = wheresWaldo.copy()
    wheresWaldoGray = cv.cvtColor(wheresWaldo, cv.COLOR_BGR2GRAY)
    
    waldo = cv.imread("./waldo.jpg")
    waldoOriginal = waldo.copy()
    waldoGray = cv.cvtColor(waldo, cv.COLOR_BGR2GRAY)
    height, width = waldoGray.shape
    
    # now we match template
    result = cv.matchTemplate(wheresWaldoGray, waldoGray, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv.rectangle(wheresWaldoOriginal, top_left, bottom_right, (0, 255, 0), 4)
    #cv.imwrite("./result8.jpg", wheresWaldoOriginal)
    
    cv.imwrite("./result8.jpg", wheresWaldoOriginal) 
    

# question 11
def doEdgeDetection():
    # read image of portrait and convert it into grayscale one
    portrait = cv.imread("./portrait.jpg")
    portraitGray = cv.cvtColor(portrait, cv.COLOR_BGR2GRAY)
    
    # do canny edge detection
    edges = cv.Canny(portrait, 140, 500)
    cv.imwrite("./result11.jpg", edges)


# main function is defined here
if __name__ == "__main__":
    # question 1
    iris = cv.imread("./iris.jpg")
    filter = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    result1 = computeCorrelation(iris, filter, "full")
    cv.imwrite("./result1.jpg", result1)
    
    # question 2
    doConvolutionUsingCorrelation(iris, 3, 5, "same")
    
    # question 7
    applyGaussionFilter()
    
    # question 8
    findWaldo()
    
    # question 11
    doEdgeDetection()
    
    
    
    
