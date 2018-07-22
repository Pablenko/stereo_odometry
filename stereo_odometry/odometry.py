import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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
        img_1 = cv.imread(f_camera_0)
        img_2 = cv.imread(f_camera_1)

        matches = match_features(img_1, img_2)
