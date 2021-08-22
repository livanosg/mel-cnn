import os
import multiprocessing as mp
import cv2
import numpy as np
import pandas as pd


def check_create_dataset(key, datasplit_path, args):
    print('Checking {} dataset\nDataset Specs: img_size: {}, colour: {}'.format(key, args['image_size'], args['colour']))
    df = pd.read_csv(datasplit_path)
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(resize_cvt_color, [(args, image_name) for image_name in df['image']])
    pool.close()
    print('Done!')


def resize_cvt_color(args, image_name):
    new_path = os.path.join(args['dir_dict']['image_folder'], image_name)
    init_path = os.path.join(args['dir_dict']['data'], image_name)

    def resize_img(input_img, size):
        rsz_rt = int(size) / max(input_img.shape)
        return cv2.resize(src=input_img, dsize=None, fx=rsz_rt, fy=rsz_rt, interpolation=cv2.INTER_NEAREST_EXACT)

    if not os.path.isfile(new_path):
        image = cv2.imread(init_path)
        image = resize_img(image, 500)  # Resize to 500pxl
        # ------------------====================== Remove hair ========================----------------------------#
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)  # Black hat filter
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)  # Gaussian filter
        ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)  # Binary thresholding (MASK)
        image = cv2.inpaint(image, mask, 6, cv2.INPAINT_NS)  # Replace pixels of the mask
        # ------------------======================== Resize ===========================----------------------------#
        if int(args['image_size']) != 500:
            image = resize_img(image, args['image_size'])
        dx = (image.shape[0] - image.shape[1]) / 2  # Compare height-width
        tblr = [np.ceil(abs(dx)), np.floor(abs(dx)), 0, 0]  # Pad top-bottom
        if dx > 0:  # If height > width
            tblr = tblr[2:] + tblr[:2]  # Pad left-right
        image = cv2.copyMakeBorder(image, *tblr, borderType=cv2.BORDER_CONSTANT)
        if args['colour'] == 'grey':
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        if not cv2.imwrite(new_path, image):  # 3-channel gray
            with open("failed_to_covert.txt", "a") as f:
                f.write('{} to new_path\n'.format(init_path, new_path))
