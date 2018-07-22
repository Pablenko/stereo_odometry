from os import listdir
from os.path import isfile, isdir, join


def files(path):
    if isdir(path):
        f_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        f_list.sort()
        return f_list
    else:
        raise ValueError("Improper path: " + path)


def parse_file(location):
    with open(location, "r") as file:
        for line in file:
            yield line
