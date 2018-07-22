from common.file_operations import files
from plotter.plot import plot_kitti
from stereo_odometry.odometry import run

import argparse


def parse_user_args():
    parser = argparse.ArgumentParser(description='Stereo odometry with kitti dataset evaluation')
    parser.add_argument('--poses', dest='poses', type=str, nargs=1, help='destination of kitti odometry poses')
    args = parser.parse_args()
    return args


def main():
    args = parse_user_args()
    if args.poses:
        files_list = files(args.poses[0])
        plot_kitti(files_list[0])


if __name__ == "__main__":
    main()
