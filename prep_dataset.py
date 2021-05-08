import multiprocessing as mp
import math
import os
import cv2.cv2 as cv2
import pandas as pd


def resize_conv_colour(image, new_dir, img_size=224, colour="grey"):
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
    path = os.path.join(new_dir, image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, src)


def check_create_dataset(data_dir):
    _, img_size, colour = os.path.basename(data_dir).split(sep="_")
    if not os.path.exists(data_dir):
        print(f"Dataset {data_dir} does not exists\n"
              f"Create dataset with Specs: img_size: {img_size}, colour: {colour}")
        os.environ["OMP_NUM_THREADS"] = "1"
        all_data = pd.read_csv('all_data_init.csv')
        images = all_data["image"]
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(resize_conv_colour, [(image, data_dir, int(img_size), colour) for image in images])
        pool.close()
        print("Done!")
    else:
        print(f"Dataset {data_dir} exists!\n"
              f"Dataset Specs: img_size: {img_size}, colour: {colour}")
