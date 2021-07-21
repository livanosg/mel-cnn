import multiprocessing as mp
import math
import os
import cv2
import pandas as pd


def resize_cvt_color(image_path, new_folder, img_size=224, colour="grey"):
    assert colour in ("grey", "rgb")
    image = cv2.imread(image_path)
    if colour == "grey":
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    dx = image.shape[0] - image.shape[1]  # Compare height-width
    if dx < 0:  # If width > height
        top, bot = math.ceil(-dx / 2), math.floor(-dx / 2)
        image = cv2.copyMakeBorder(image, top, bot, 0, 0, borderType=cv2.BORDER_CONSTANT)
    else:  # If width < height
        left, right = math.ceil(dx / 2), math.floor(dx / 2)
        image = cv2.copyMakeBorder(image, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT)
    image = cv2.resize(src=image, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST_EXACT)
    new_path = os.path.join(new_folder, image_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    cv2.imwrite(new_path, image)


def check_create_dataset(img_size, colour, dir_dict, force=False):
    os.environ["OMP_NUM_THREADS"] = "1"
    if not os.path.exists(dir_dict["image_folder"]) or force is True:
        if not force:
            log_info = "Dataset {} does not exists\nCreate dataset with Specs: img_size: {}, colour: {}"
        else:
            log_info = "Overwriting dataset in {}\nDataset Specs: img_size: {}, colour: {}"
        print(log_info.format(dir_dict['image_folder'], img_size, colour))
        features = pd.read_csv('all_data.csv')
        features["image"] = "data/" + features["dataset_id"] + "/data/" + features["image"]

        if mp.cpu_count() > 32:
            threads = 32
        else:
            threads = mp.cpu_count()
        pool = mp.Pool(threads)
        pool.starmap(resize_cvt_color, [(image_path, dir_dict['image_folder'], int(img_size), colour) for image_path in features["image"]])
        pool.close()
        print("Done!")
    else:
        log_info = "Dataset {} exists!\nDataset Specs: img_size: {}, colour: {}"
        print(log_info.format(dir_dict['image_folder'], img_size, colour))


if __name__ == '__main__':
    exit()
