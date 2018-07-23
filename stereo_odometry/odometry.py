import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def create_disparity_map(img_1, img_2):
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_1, img_2)
    return disparity


def klt_tracking(img_1, img_2):
    corners_img_1 = cv.goodFeaturesToTrack(img_1, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    corners_img_2, status, errors = cv.calcOpticalFlowPyrLK(img_1, img_2, corners_img_1, None, winSize=(15, 15), maxLevel=2, \
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    good_corners_img_1 = corners_img_1[status == 1]
    good_corners_img_2 = corners_img_2[status == 1]


def match_features(img_1, img_2):
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp_1, des_1 = orb.detectAndCompute(img_1, None)
    kp_2, des_2 = orb.detectAndCompute(img_2, None)

    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key = lambda x : x.distance)

    img_3 = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches[:10], None)
    plt.imshow(img_3),plt.show()

    return matches


def stereo_odometry(files_camera_0, files_camera_1):
    for f_camera_0, f_camera_1 in zip(files_camera_0[0:2], files_camera_1[0:2]):
        img_1 = cv.imread(f_camera_0, cv.IMREAD_GRAYSCALE)
        img_2 = cv.imread(f_camera_1, cv.IMREAD_GRAYSCALE)

        disparity_map = create_disparity_map(img_1, img_2)
        klt_tracking(img_1, img_2)
