import multiprocessing as mp
import math
import os
import cv2
import pandas as pd

from config import IMAGE_FOLDER


def resize_conv_colour(image, img_size=224, colour="grey"):
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
    path = os.path.join(IMAGE_FOLDER.format(str(img_size), colour), image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, src)


def check_create_dataset(img_size, colour, dir_dict, force=False):
    os.environ["OMP_NUM_THREADS"] = "1"
    if not os.path.exists(dir_dict["image_folder"]) or force is True:
        if not force:
            log_info = "Dataset {} does not exists\nCreate dataset with Specs: img_size: {}, colour: {}"
        else:
            log_info = "Overwriting dataset in {}\nDataset Specs: img_size: {}, colour: {}"
        print(log_info.format(dir_dict['image_folder'], img_size, colour))
        all_data = pd.read_csv('all_data_init.csv')
        if mp.cpu_count() > 32:
            threads = 32
        else:
            threads = mp.cpu_count()
        pool = mp.Pool(threads)
        pool.starmap(resize_conv_colour, [(image, int(img_size), colour) for image in all_data["image"]])
        pool.close()
        print("Done!")
    else:
        log_info = "Dataset {} exists!\nDataset Specs: img_size: {}, colour: {}"
        print(log_info.format(dir_dict['image_folder'], img_size, colour))


if __name__ == '__main__':
    exit()
