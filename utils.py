import numpy as np
from PIL import Image


def show_image(img: np.ndarray):
    Image.fromarray(img).show()


def save_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def save_crop(img: np.ndarray, crop: tuple, path: str):
    # (w0, h0, w1, h1)
    path += f'_{crop[0]}x{crop[1]}_{crop[2]}x{crop[3]}.png'
    Image.fromarray(img[crop[1]:crop[3], crop[0]:crop[2]]).save(path)


def crop_image(img: np.ndarray, crop: tuple):
    return img[crop[1]:crop[3], crop[0]:crop[2]]
