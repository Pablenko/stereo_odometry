from collections import namedtuple
from common.file_operations import parse_file
from sys import maxint

import matplotlib.pyplot as plt
import numpy as np


ROWS = 3
COLS = 4
STEP_SIZE = 10
MAX_INT = maxint
MIN_INT = -maxint - 1

axes_range = namedtuple('axes_range', 'x_min, x_max, y_min, y_max')


def get_floats(string):
    return [float(x) for x in string.split()]


def create_matrix(f_array, n_rows, n_cols):
    def chunks():
        for i in xrange(n_rows):
            yield f_array[i*n_cols:(i+1)*n_cols]
    return np.matrix([col for col in chunks()])


def calculate_plot_axes(poses):
    x_min = MAX_INT
    y_min = MAX_INT
    x_max = MIN_INT
    y_max = MIN_INT

    for m in poses:
        x = m.item(0, 3)
        y = m.item(2, 3)

        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    x_min *= 1.2
    x_max *= 1.2
    y_min *= 1.2
    y_max *= 1.2

    return axes_range(x_min, x_max, y_min, y_max)


def transform_poses_to_2d_plane(poses):
    size = len(poses) / STEP_SIZE
    x = np.zeros(size)
    y = np.zeros(size)

    for i in range(0, size):
        x[i] = poses[i*STEP_SIZE].item(0, 3)
        y[i] = poses[i*STEP_SIZE].item(2, 3)

    return x, y


def plot_vehicle_path(x_min, x_max, y_min, y_max, poses_arr):
    for p in poses_arr:
        x, y = transform_poses_to_2d_plane(p)
        axes_sizing = calculate_plot_axes(p)
        axes = plt.gca()
        axes.set_xlim([axes_sizing.x_min, axes_sizing.x_max])
        axes.set_ylim([axes_sizing.y_min, axes_sizing.y_max])

        plt.plot(x, y)
        plt.show()


def plot_kitti(pose_file_location):
    test_data_poses = [create_matrix(get_floats(line), ROWS, COLS) for line in parse_file(pose_file_location)]
    axes_pos = calculate_plot_axes(test_data_poses)
    plot_vehicle_path(axes_pos.x_min, axes_pos.x_max, axes_pos.y_min, axes_pos.y_max, [test_data_poses])
