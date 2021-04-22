import multiprocessing as mp
import math
import os

import cv2.cv2 as cv2
import pandas as pd


def crop(image, size=224):
    src = cv2.cvtColor(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    dx = src.shape[0] - src.shape[1]
    if dx < 0:
        dx = -dx
        top, bot = math.ceil(dx / 2), math.floor(dx / 2)
        src = cv2.copyMakeBorder(src, top, bot, 0, 0, borderType=cv2.BORDER_CONSTANT)
    else:
        left, right = math.ceil(dx / 2), math.floor(dx / 2)
        src = cv2.copyMakeBorder(src, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT)
    src = cv2.resize(src=src, dsize=(size, size), interpolation=cv2.INTER_NEAREST_EXACT)
    path = f"proc_{size}/" + image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(f"proc_{size}/" + image, src)


all_data = pd.read_csv('all_data_v2.csv')
images = all_data["image"]
pool = mp.Pool(mp.cpu_count())
SIZE = 500
pool.starmap(crop, [(image, SIZE) for image in images])
pool.close()
# dst = cv.equalizeHist(src3)
# cv.imshow('Equalized Image 3', dst)
