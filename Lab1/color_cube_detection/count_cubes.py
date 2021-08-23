import cv2
import numpy as np

#TODO: Modify these values for yellow color range. Add thresholds for detecting green also.
yellow_lower = np.array([10, 130, 130])
yellow_upper = np.array([20, 255, 255])

green_lower = np.array([15, 40, 40])
green_upper = np.array([40, 120, 120])


#TODO: Change this function so that it filters the image based on color using the hsv range for each color.
def filter_image(img, hsv_lower, hsv_upper):
    # set img
    IRGB = cv2.add(img, np.array([40.0]))
    IRGB = cv2.medianBlur(IRGB, 25)
    IRGB = cv2.GaussianBlur(IRGB, (5, 5), 10)
    IRGB = cv2.bilateralFilter(IRGB, 45, 100, 100)
    
    # convert to HSV color space
    IHSV = cv2.cvtColor(IRGB, cv2.COLOR_BGR2HSV)

    # Modify mask
    # take in HSV limits for mask
    mask = cv2.inRange(IHSV, hsv_lower, hsv_upper)
    
    # debugging
    cv2.imshow("color mask", mask) 
    cv2.imshow("original", img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return mask
    
#TODO: Change the parameters to make blob detection more accurate. Hint: You might need to set some parameters to specify features such as color, size, and shape. The features have to be selected based on the application. 
def detect_blob(mask):
    img = mask
    
   # Set up the SimpleBlobdetector with default parameters with specific values.
    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByColor = False

    params.blobColor = 256

    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByInertia = False

    #ADD CODE HERE

    # builds a blob detector with the given parameters 
    detector = cv2.SimpleBlobDetector_create(params)

    # use the detector to detect blobs.
    keypoints = detector.detect(img)
    
    # debugging
    # img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("edge detector", img_with_keypoints)
    # cv2.imshow("original", img)  
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return len(keypoints)

    
def count_cubes(img):
    mask_yellow = filter_image(img, yellow_lower, yellow_upper)
    num_yellow = detect_blob(mask_yellow)
    
    #print("number of yellow is ", num_yellow)
    
    mask_green = filter_image(img, green_lower, green_upper)
    num_green = detect_blob(mask_green)

    #print("number of green is ", num_green)    

    # debugging 
    # cv2.imshow("edge detector", mask_green)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #TODO: Modify to return number of detected cubes for both yellow and green (instead of 0)
    return num_yellow, num_green

