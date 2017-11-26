"""
Util functions
"""

import numpy as np
from PIL import Image


def load_data(imagefilepath):
    """
    loading image file from filepath
    input: imagefilepath
    return: image obj (dimension is one)
    """
    img_obj = Image.open(imagefilepath).convert("L")
    img = np.array(img_obj, dtype=np.float32)
    img = img * 1.0 / 255
    width, height = img.shape
    img = img.reshape(width * height)
    return img


def load_label(labelfilepath):
    """
    loading label
    """
    with open(labelfilepath, "r") as readfile:
        labellist = [line.strip() for line in readfile.readlines()]
    return labellist
