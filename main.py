from common.file_operations import files, filter_image_files
from plotter.plot import plot_against_kitti, parse_ground_truth_poses, report_errors
from stereo_odometry.odometry import stereo_odometry, calculate_errors

import argparse
import os


IMAGE_O_DIR = 'image_0'
IMAGE_1_DIR = 'image_1'


def parse_user_args():
    parser = argparse.ArgumentParser(description='Stereo odometry with kitti dataset evaluation')
    parser.add_argument('--poses', dest='poses', type=str, nargs=1, required=True, help='destination of kitti odometry poses file')
    parser.add_argument('--data', dest='data', type=str, nargs=1, required=True, help='destination of input images')
    parser.add_argument('--limit', dest='limit', type=int, nargs=1, help='limit number of processed images')
    args = parser.parse_args()
    return args


def main():
    args = parse_user_args()
    poses_file= args.poses[0]
    kitti_data_path = args.data[0]

    path_camera_0 = os.path.join(kitti_data_path, IMAGE_O_DIR)
    path_camera_1 = os.path.join(kitti_data_path, IMAGE_1_DIR)
    files_camera_0 = sorted(filter_image_files(files(path_camera_0)))
    files_camera_1 = sorted(filter_image_files(files(path_camera_1)))

    limit = args.limit[0] if args.limit else len(files_camera_0)
    translations = [tr for tr in stereo_odometry(files_camera_0[0:limit], files_camera_1[0:limit])]
    ground_truth_poses = parse_ground_truth_poses(poses_file)
    errors, max_error = calculate_errors(translations, ground_truth_poses[0:limit])

    report_errors(errors, max_error)
    plot_against_kitti(translations, ground_truth_poses)


if __name__ == "__main__":
    main()
