import argparse

from stereo_odometry.odometry import run
from plotter.plot import plot_kitti


def parse_user_args():
    parser = argparse.ArgumentParser(description='Stereo odometry with kitti dataset evaluation')

    parser.add_argument('--poses', dest='poses', type=str, nargs=1, help='destination of kitti odometry poses')

    args = parser.parse_args()

    return args

def main():
    args = parse_user_args()
    if args.poses:
        plot_kitti(args.poses)

if __name__ == "__main__":
    main()
