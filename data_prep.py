import os
import multiprocessing as mp
import math
import cv2
import pandas as pd


def resize_cvt_color(sample, args):
    image_path = os.path.join(args['dir_dict']['data'], sample['image'])
    new_path = os.path.join(args['dir_dict']['image_folder'], sample['image'])
    if not os.path.isfile(new_path):
        try:
            image = cv2.imread(image_path)
            resize = min(int(args['image_size']) / image.shape[0],  int(args['image_size']) / image.shape[1])
            image = cv2.resize(src=image, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST_EXACT)
            dx = (image.shape[0] - image.shape[1]) / 2  # Compare height-width
            tblr = [math.ceil(math.fabs(dx)), math.floor(math.fabs(dx)), 0, 0]  # Pad top-bottom
            if dx > 0:  # If height > width
                tblr = tblr[2:] + tblr[:2]  # Pad left-right
            image = cv2.copyMakeBorder(image, *tblr, borderType=cv2.BORDER_CONSTANT)
            if args['colour'] == 'grey':
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)  # 3-channel gray
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            cv2.imwrite(new_path, image)
        except:
            with open("failed_to_covert.txt", "a") as f:
                f.write(f"{image_path} 'to '{new_path}'\n'")


def check_create_dataset(args, force=False):
    os.environ['OMP_NUM_THREADS'] = '1'
    if not os.path.exists(args['dir_dict']['image_folder']) or force is True:
        print(f"Writing dataset in {args['dir_dict']['image_folder']}\nDataset Specs: img_size: {args['image_size']}, colour: {args['colour']}")
        samples = pd.read_csv(args['dir_dict']['data_csv']['train']).append(
            pd.read_csv(args['dir_dict']['data_csv']['val'])).append(
            pd.read_csv(args['dir_dict']['data_csv']['test'])).append(
            pd.read_csv(args['dir_dict']['data_csv']['isic_20_test']))
        pool = mp.Pool(len(os.sched_getaffinity(0)))
        pool.starmap(resize_cvt_color, [(sample, args) for _, sample in samples.iterrows()])
        pool.close()
        print('Done!')
    else:
        print(f"Dataset {args['dir_dict']['image_folder']} exists!")
