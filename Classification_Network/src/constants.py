import os

dir_data = "../data/"
dir_outputs = "../outputs/"
dir_models = "../models"
dir_cropped_Images = "../outputs/cropped_Images"

ts_fmt = "%Y%m%d_%H%M%S"


def make_dirs():
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)
    if not os.path.exists(dir_outputs):
        os.makedirs(dir_outputs)
    if not os.path.exists(dir_models):
        os.makedirs(dir_models)
    if not os.path.exists(dir_cropped_Images):
        os.makedirs(dir_cropped_Images)


make_dirs()
