from os import listdir
from os.path import isfile, isdir, join


IMG_FILES = ['png', 'jpg', 'jpeg']


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


def filter_image_files(files):
    def filt():
        for f in files:
            for ext in IMG_FILES:
                if ext in f:
                    yield f

    return [f for f in filt()]
