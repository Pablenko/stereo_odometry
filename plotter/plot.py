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


def transform_translations_to_2d_plane(translations):
    x = np.zeros(len(translations))
    y = np.zeros(len(translations))

    for i, tr in enumerate(translations):
        x[i] = tr[0]
        y[i] = tr[2]

    return x, y


def plot_vehicle_path(x_min, x_max, y_min, y_max, translations, poses):
    p_x, p_y = transform_poses_to_2d_plane(poses)
    axes_sizing = calculate_plot_axes(poses)

    tr_x, tr_y = transform_translations_to_2d_plane(translations)

    axes = plt.gca()
    axes.set_xlim([axes_sizing.x_min, axes_sizing.x_max])
    axes.set_ylim([axes_sizing.y_min, axes_sizing.y_max])

    plt.plot(p_x, p_y)
    plt.plot(tr_x, tr_y)
    plt.show()


def parse_ground_truth_poses(file_path):
    return [create_matrix(get_floats(line), ROWS, COLS) for line in parse_file(file_path)]


def plot_against_kitti(translations, ground_truth_poses):
    axes_pos = calculate_plot_axes(ground_truth_poses)
    plot_vehicle_path(axes_pos.x_min, axes_pos.x_max, axes_pos.y_min, axes_pos.y_max, translations, ground_truth_poses)


def report_errors(errors, max_error):
    print 'Translation erros (against ground truth poses):'

    for i, e in enumerate(errors):
        print 'Num: ' + str(i+1) + ', error: ' + str(e)

    print 'Maximal error: ' + str(max_error)
