from os import listdir
from os.path import isfile, isdir, join

import matplotlib.pyplot as plt
import numpy as np


ROWS = 3
COLS = 4
STEP_SIZE = 10


def files(poses_path):
    if isdir(poses_path):
        f_list = [join(poses_path, f) for f in listdir(poses_path) if isfile(join(poses_path, f))]
        f_list.sort()
        return f_list
    else:
        raise ValueError("Improper kitti poses path: " + poses_path)


def get_floats(string):
    return [float(x) for x in string.split()]


def create_matrix(f_array, n_rows, n_cols):
    def chunks():
        for i in xrange(n_rows):
            yield f_array[i*n_cols:(i+1)*n_cols]
    return np.matrix([col for col in chunks()])


def parse_poses_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield create_matrix(get_floats(line), ROWS, COLS)


def calculate_plot_axes(poses):
    return -400, 400, -100, 700


def transform_poses_to_2d_plane(poses):
    size = len(poses) / STEP_SIZE
    x = np.zeros(size)
    y = np.zeros(size)

    for i in range(0, size):
        x[i] = poses[i*STEP_SIZE].item(0, 3)
        y[i] = poses[i*STEP_SIZE].item(2, 3)

    return x, y


def plot_path(x_min, x_max, y_min, y_max, poses_arr):
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])

    for p in poses_arr:
        x, y = transform_poses_to_2d_plane(p)
        plt.plot(x, y)

    plt.show()


def plot_kitti(poses_path):
    files_list = files(poses_path[0])
    for f in files_list[0:1]:
        test_data_poses = [x for x in parse_poses_file(f)]
        x_min, x_max, y_min, y_max = calculate_plot_axes(test_data_poses)
        plot_path(x_min, x_max, y_min, y_max, [test_data_poses])
