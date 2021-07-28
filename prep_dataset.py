import os
import multiprocessing as mp
import math
import cv2
import pandas as pd


def resize_cvt_color(sample, args):
    image_path = os.path.join(args['dir_dict']['data'], sample['dataset_id'], 'data', sample['image'])
    new_path = os.path.join(args['dir_dict']['image_folder'], 'data', sample['dataset_id'], 'data', sample['image'])
    if os.path.isfile(new_path):
        pass
    else:
        print(f"{image_path} does not exist")
        image = cv2.imread(image_path)
        if args['colour'] == 'grey':
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        try:
            dx = image.shape[0] - image.shape[1]  # Compare height-width
            if dx < 0:  # If width > height
                top, bot = math.ceil(-dx / 2), math.floor(-dx / 2)
                image = cv2.copyMakeBorder(image, top, bot, 0, 0, borderType=cv2.BORDER_CONSTANT)
            else:
                left, right = math.ceil(dx / 2), math.floor(dx / 2)
                image = cv2.copyMakeBorder(image, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT)
            image = cv2.resize(src=image, dsize=(int(args['image_size']), int(args['image_size'])),
                               interpolation=cv2.INTER_NEAREST_EXACT)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            cv2.imwrite(new_path, image)
        except AttributeError:
            with open("failed_to_covert.txt", "a") as f:
                f.write(image_path + '\n')


def check_create_dataset(args, force=False):
    os.environ['OMP_NUM_THREADS'] = '1'
    if not os.path.exists(args['dir_dict']['image_folder']) or force is True:
        if force:
            log_info = 'Overwriting dataset in {}\nDataset Specs: img_size: {}, colour: {}'
        else:
            log_info = 'Dataset {} does not exists\nCreate dataset with Specs: img_size: {}, colour: {}'
        print(log_info.format(args['dir_dict']['image_folder'], args['image_size'], args['colour']))
        samples = pd.read_csv(args['dir_dict']['data_csv']['train']).append(pd.read_csv(args['dir_dict']['data_csv']['val'])).append(pd.read_csv(args['dir_dict']['data_csv']['test']))
        if mp.cpu_count() > 32:
            threads = 32
        else:
            threads = mp.cpu_count()
        print(f"Number of threads: {threads}.")
        pool = mp.Pool(threads)
        pool.starmap(resize_cvt_color, [(sample, args) for _, sample in samples.iterrows()])
        pool.close()
        print('Done!')

    else:
        log_info = 'Dataset {} exists!\nDataset Specs: img_size: {}, colour: {}'
        print(log_info.format(args['dir_dict']['image_folder'], args['image_size'], args['colour']))


if __name__ == '__main__':
    pass
