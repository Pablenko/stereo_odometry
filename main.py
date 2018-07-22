from common.file_operations import files, filter_image_files
from plotter.plot import plot_kitti
from stereo_odometry.odometry import stereo_odometry

import argparse
import os


IMAGE_O_DIR = 'image_0'
IMAGE_1_DIR = 'image_1'


def parse_user_args():
    parser = argparse.ArgumentParser(description='Stereo odometry with kitti dataset evaluation')
    parser.add_argument('--poses', dest='poses', type=str, nargs=1, help='destination of kitti odometry poses')
    parser.add_argument('--data', dest='data', type=str, nargs=1, help='destination of input images')
    args = parser.parse_args()
    return args


def main():
    args = parse_user_args()
    if args.poses:
        files_list = files(args.poses[0])
        plot_kitti(files_list[0])
    if args.data:
        kitti_data_path = args.data[0]
        path_camera_0 = os.path.join(kitti_data_path, IMAGE_O_DIR)
        path_camera_1 = os.path.join(kitti_data_path, IMAGE_1_DIR)
        files_camera_0 = sorted(filter_image_files(files(path_camera_0)))
        files_camera_1 = sorted(filter_image_files(files(path_camera_1)))
        stereo_odometry(files_camera_0, files_camera_1)


if __name__ == "__main__":
    main()
