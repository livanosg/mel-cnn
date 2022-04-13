import os
import cv2
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def setup_images(args, dirs):
    print('Setting up Datasets...')
    for data_set in ('train', 'validation', 'test', 'isic20_test'):
        df = pd.read_csv(dirs[data_set])
        Parallel(n_jobs=16)(delayed(hair_removal_and_resize)(image_name, args, dirs) for (image_name) in df['image'])
    print('Done!')


def hair_removal_and_resize(image_name, args, dirs):
    def resize_img(image_to_resize, size):
        size_ratio = int(size) / max(np.shape(image_to_resize)[:-1])
        return cv2.resize(src=image_to_resize, dsize=None,
                          fx=size_ratio, fy=size_ratio, interpolation=cv2.INTER_NEAREST_EXACT)

    def hair_removal(image_to_remove_hair):
        gray_scale = cv2.cvtColor(image_to_remove_hair, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (9, 9))
        blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)  # Black hat filter
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)  # Gaussian filter
        ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)  # Binary thresholding (MASK)
        return cv2.inpaint(image_to_remove_hair, mask, 6, cv2.INPAINT_NS)  # Replace pixels of the mask

    if not os.path.isfile(os.path.join(dirs['proc_img_folder'], image_name)):
        image = cv2.imread(os.path.join(dirs['init_img_folder'], image_name))
        image = resize_img(image, 500)  # Resize to 500pxl for faster processing
        image = hair_removal(image)
        # if args['image_size'] != 500:
        image = resize_img(image, args['image_size'])
        dx = (image.shape[0] - image.shape[1]) / 2  # Compare height-width
        tblr = [int(np.ceil(np.abs(dx))), int(np.floor(np.abs(dx))), 0, 0]  # Pad top-bottom
        if dx > 0:  # If height > width
            tblr = tblr[2:] + tblr[:2]  # Pad left-right
        image = cv2.copyMakeBorder(image, *tblr, borderType=cv2.BORDER_CONSTANT)  # Pad with zeros to make it squared.
        new_path = os.path.join(dirs['proc_img_folder'], image_name)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        if not cv2.imwrite(new_path, image):
            with open('fail_to_save.txt', 'a+') as f:
                f.write('{}\n'.format(new_path))
