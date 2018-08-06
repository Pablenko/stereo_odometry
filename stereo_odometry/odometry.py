import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


FLANN_INDEX_KDTREE = 1


def klt_tracking(img_1, img_2, corners_img_1, world_points):
    corners_img_2, status, errors = cv.calcOpticalFlowPyrLK(img_1, img_2, corners_img_1, None, winSize=(21, 21), maxLevel=3, \
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    status = status.reshape(status.shape[0])
    good_corners_img_1 = corners_img_1[status == 1]
    good_corners_img_2 = corners_img_2[status == 1]
    world_points = world_points[status == 1]

    return good_corners_img_1, good_corners_img_2, world_points


#TODO: read it from somewhere
def get_k_matrix():
    return np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                     [0, 7.188560000000e+02, 1.852157000000e+02],
                     [0, 0, 1]])


def remove_duplicate(query_points, ref_points, radius=5):
    for i in range(len(query_points)):
        query = query_points[i]
        xliml, xlimh = query[0]-radius, query[0]+radius
        yliml, ylimh = query[1]-radius, query[1]+radius
        inside_x_lim_mask = (ref_points[:,0] > xliml) & (ref_points[:,0] < xlimh)
        curr_kps_in_x_lim = ref_points[inside_x_lim_mask]

        if curr_kps_in_x_lim.shape[0] != 0:
            inside_y_lim_mask = (curr_kps_in_x_lim[:,1] > yliml) & (curr_kps_in_x_lim[:,1] < ylimh)
            curr_kps_in_x_lim_and_y_lim = curr_kps_in_x_lim[inside_y_lim_mask,:]
            if curr_kps_in_x_lim_and_y_lim.shape[0] != 0:
                query_points[i] =  np.array([0,0])
    return (query_points[:, 0]  != 0 )


def extract_keypoints_surf(img1, img2, K, baseline, ref_points = None):
    detector = cv.xfeatures2d.SURF_create(400)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2,None)

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)

    match_points1, match_points2 = [], []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            match_points1.append(kp1[m.queryIdx].pt)
            match_points2.append(kp2[m.trainIdx].pt)

    p1 = np.array(match_points1).astype(float)
    p2 = np.array(match_points2).astype(float)

    if ref_points is not None:
        mask = remove_duplicate(p1, ref_points)
        p1 = p1[mask,:]
        p2 = p2[mask,:]

    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline,0, 0]]).T)))

    p1_flip = np.vstack((p1.T,np.ones((1,p1.shape[0]))))
    p2_flip = np.vstack((p2.T,np.ones((1,p2.shape[0]))))

    P = cv.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    P = P/P[3]
    land_points = P[:3]

    return land_points.T, p1


def stereo_odometry(files_left_camera, files_right_camera):
    k_matrix = get_k_matrix()
    baseline = 0.54 #TODO: check from where it comes

    reference_img = cv.imread(files_left_camera[0], cv.IMREAD_GRAYSCALE)
    first_right_img = cv.imread(files_right_camera[0], cv.IMREAD_GRAYSCALE)

    landmark_3d, reference_2d = extract_keypoints_surf(reference_img, first_right_img, k_matrix, baseline)
    reference_2d = reference_2d.astype('float32')

    for i, (left_image_loc, right_image_loc) in enumerate(zip(files_left_camera, files_right_camera)):
        print 'Step number: ' + str(i)

        left_image = cv.imread(left_image_loc, cv.IMREAD_GRAYSCALE)
        right_image = cv.imread(right_image_loc, cv.IMREAD_GRAYSCALE)

        reference_2d, tracked_2d_points, landmark_3d = klt_tracking(reference_img, left_image, reference_2d, landmark_3d)
        pnp_obj = np.expand_dims(landmark_3d, axis = 2)
        pnp_cur = np.expand_dims(tracked_2d_points, axis = 2).astype(float)

        _, rvec, tvec, inliers = cv.solvePnPRansac(pnp_obj, pnp_cur, k_matrix, None)

        reference_2d = tracked_2d_points[inliers[:,0],:]
        landmark_3d  = landmark_3d[inliers[:,0],:]

        rot_matrix, _ = cv.Rodrigues(rvec)
        tvec = -rot_matrix.T.dot(tvec)

        inv_transform = np.hstack((rot_matrix.T, tvec))
        inliers_ratio = len(inliers) / len(pnp_obj)

        if inliers_ratio < 0.9 or (len(reference_2D) < 50):
            new_landmark_3d, new_reference_2d  = extract_keypoints_surf(left_image, right_image, k_matrix, baseline, reference_2d)
            new_reference_2d = new_reference_2d.astype('float32')
            new_landmark_3d = inv_transform.dot(np.vstack((new_landmark_3d.T, np.ones((1, new_landmark_3d.shape[0])))))
            valid_matches = new_landmark_3d[2,:] > 0
            new_landmark_3d = new_landmark_3d[:,valid_matches]

            reference_2d = np.vstack((reference_2d, new_reference_2d[valid_matches,:]))
            landmark_3d =  np.vstack((landmark_3d, new_landmark_3d.T))

        reference_img = left_image

        print 'Result translation vector: '
        print tvec

        yield tvec


def calculate_errors(translations, reference):
    errors = []
    max_error = 0

    for tr, ref in zip(translations, reference):
        error = np.sqrt((tr[0]-ref.item(0, 3))**2 + (tr[1]-ref.item(1, 3))**2 + (tr[2]-ref.item(2, 3))**2)
        max_error = error if error > max_error else max_error
        errors.append(error)

    return errors, max_error
