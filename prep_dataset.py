import multiprocessing as mp
import math
import os
import cv2.cv2 as cv2
import pandas as pd


def resize_conv_colour(image, folder_format, img_size=224, colour="grey"):
    assert colour in ("grey", "rgb")
    src = cv2.imread(image)
    if colour == "grey":
        src = cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    dx = src.shape[0] - src.shape[1]
    if dx < 0:
        top, bot = math.ceil(-dx / 2), math.floor(-dx / 2)
        src = cv2.copyMakeBorder(src, top, bot, 0, 0, borderType=cv2.BORDER_CONSTANT)
    else:
        left, right = math.ceil(dx / 2), math.floor(dx / 2)
        src = cv2.copyMakeBorder(src, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT)
    src = cv2.resize(src=src, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST_EXACT)
    path = folder_format.format(img_size, colour) + image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, src)


def create_dataset(img_size, colour, folder_format):
    os.environ["OMP_NUM_THREADS"] = "1"
    all_data = pd.read_csv('all_data_init.csv')
    images = all_data["image"]
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(resize_conv_colour, [(image, folder_format, img_size, colour) for image in images])
    pool.close()
