from os import listdir
from os.path import isfile, isdir, join

import matplotlib.pyplot as plt
import numpy as np


def files(poses_path):
    if isdir(poses_path):
        f_list = [join(poses_path, f) for f in listdir(poses_path) if isfile(join(poses_path, f))]
        f_list.sort()
        return f_list
    else:
        raise ValueError("Improper kitti poses path: " + poses_path)


def get_floats(string):
    return [float(x) for x in string.split()]


def parse_poses_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield get_floats(line)


def plot_path():
    t = np.arange(0.0, 1.0, 0.01)
    for n in [1, 2]:
        plt.plot(t, t**n)

    plt.show()


def plot_kitti(poses_path):
    files_list = files(poses_path[0])
    for f in files_list[0:1]:
        for num, data in enumerate(parse_poses_file(f)):
            print num
            print data

