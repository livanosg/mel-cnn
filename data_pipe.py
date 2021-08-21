import os
import multiprocessing as mp
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from config import DATA_MAP, BEN_MAL_MAP, NEV_MEL_MAP, NP_RNG


class MelData:
    def __init__(self, args: dict):
        self.args = args
        self.TF_RNG = tf.random.Generator.from_seed(1312)  # .from_non_deterministic_state()
        self.seeds = self.TF_RNG.make_seeds(5)
        self.batch_size = self.args['batch_size'] * self.args['replicas']
        self.data_df = {'train': pd.read_csv(self.args['dir_dict']['data_csv']['train']).sample(frac=self.args['dataset_frac'], random_state=NP_RNG.bit_generator),
                        'val': pd.read_csv(self.args['dir_dict']['data_csv']['val']).sample(frac=1., random_state=NP_RNG.bit_generator),
                        'test': pd.read_csv(self.args['dir_dict']['data_csv']['test']),
                        'isic20_test': pd.read_csv(self.args['dir_dict']['data_csv']['isic20_test'])}
        self.class_counts = dict(self.data_df['train']['class'].value_counts())
        self.image_type_counts = dict(self.data_df['train']['image_type'].value_counts())
        self.weights_per_class = len(self.data_df['train']) / np.asarray(self.args['num_classes'] * [self.class_counts[k] for k in sorted(self.class_counts)])
        self.weights_per_image_type = len(self.data_df['train']) / np.asarray(len(self.image_type_counts) * [self.image_type_counts[k] for k in sorted(self.image_type_counts)])
        self.weights_per_image_type = np.sqrt(self.weights_per_image_type)  # Through sqrt

    def resize_cvt_color(self, image_name):
        new_path = os.path.join(self.args['dir_dict']['image_folder'], image_name)
        init_path = os.path.join(self.args['dir_dict']['data'],  image_name)
        def resize(image, size):
            resize = int(size) / max(image.shape)
            return cv2.resize(src=image, dsize=None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST_EXACT)

        if not os.path.isfile(new_path):
            image = cv2.imread(init_path)
            image = resize(image, 500)  # Resize to 500pxl
            # ------------------====================== Remove hair ========================----------------------------#
            grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(1, (9, 9))
            blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)  # Black hat filter
            bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)  # Gaussian filter
            ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)  # Binary thresholding (MASK)
            image = cv2.inpaint(image, mask, 6, cv2.INPAINT_NS)  # Replace pixels of the mask
            # ------------------======================== Resize ===========================----------------------------#
            if int(self.args['image_size']) != 500:
                image = resize(image, self.args['image_size'])
            dx = (image.shape[0] - image.shape[1]) / 2  # Compare height-width
            tblr = [np.ceil(abs(dx)), np.floor(abs(dx)), 0, 0]  # Pad top-bottom
            if dx > 0:  # If height > width
                tblr = tblr[2:] + tblr[:2]  # Pad left-right
            image = cv2.copyMakeBorder(image, *tblr, borderType=cv2.BORDER_CONSTANT)
            if self.args['colour'] == 'grey':
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            if not cv2.imwrite(new_path, image):  # 3-channel gray
                with open("failed_to_covert.txt", "a") as f:
                    f.write(f"{init_path} 'to '{new_path}'\n'")

    def check_create_dataset(self, image_df):
        os.environ['OMP_NUM_THREADS'] = '1'
        print(f"Checking dataset in {self.args['dir_dict']['image_folder']}\nDataset Specs: img_size: {self.args['image_size']}, colour: {self.args['colour']}")
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(self.resize_cvt_color, [(image_name, self.args) for image_name in image_df])
        pool.close()
        print('Done!')

    def prep_data(self, df: pd.DataFrame, mode):
        self.check_create_dataset(df['image'])
        df['image'] = df['image'].apply(lambda x: os.path.join(self.args['dir_dict']['image_folder'], x))
        if self.args['task'] == 'ben_mal':
            mapper = BEN_MAL_MAP
        elif self.args['task'] == 'nev_mel':
            mapper = NEV_MEL_MAP
        else:
            mapper = None
        df = df.replace(to_replace=mapper)
        if self.args['image_type'] != 'both':  # Keep derm or clinic, samples.
            df.drop(df[df['image_type'] != DATA_MAP['image_type'][self.args['image_type']]].index, errors='ignore', inplace=True)
        if mode != 'isic20_test':
            if self.args['task'] == 'nev_mel':  # Drop : NNV, NMC, SUS, unknown
                df.drop(df[df['class'] == 2].index, errors='ignore', inplace=True)
            if self.args['task'] == '5cls':  # Drop: unknown
                df.drop(df[df['class'] == 5].index, errors='ignore', inplace=True)
        ohe_features = {'image_path': tf.convert_to_tensor(df['image'])}
        if not self.args['only_image']:
            if not self.args['no_image_type']:
                ohe_features['image_type'] = tf.keras.backend.one_hot(indices=df['image_type'], num_classes=2)
            ohe_features['sex'] = tf.keras.backend.one_hot(indices=df['sex'], num_classes=2)
            ohe_features['age_approx'] = tf.keras.backend.one_hot(indices=df['age_approx'], num_classes=10)
            ohe_features['location'] = tf.keras.backend.one_hot(indices=df['location'], num_classes=6)
        labels = None
        df['sample_weights'] = 1.
        if mode != 'isic20_test':
            if not self.args['no_image_weights']:
                for idx1, _class in enumerate(sorted(self.class_counts)):
                    for idx2, image_type in enumerate(sorted(self.image_type_counts)):
                        df.loc[(df['image_type'] == image_type) & (df['class'] == _class), 'sample_weights'] = self.weights_per_image_type[idx2]  # + self.weights_per_class[idx1]
                df['sample_weights'] /= df['sample_weights'].min()
            labels = {'class': tf.keras.backend.one_hot(indices=df['class'], num_classes=self.args['num_classes'])}
        sample_weights = tf.convert_to_tensor(df['sample_weights'].astype(float))
        return ohe_features, labels, sample_weights

    def get_dataset(self, pick_dataset=None, repeat=1):
        data_dict = {}
        for data_name in ['train', 'val', 'test', 'isic20_test']:
            data_dict[data_name] = self.prep_data(self.data_df[data_name], mode=data_name)
        data = data_dict[pick_dataset]

        def tf_imread(sample, label=None, sample_weight=None):
            sample['image'] = tf.image.decode_image(tf.io.read_file(sample['image_path']), channels=3, dtype=tf.uint8)
            sample['image'] = tf.reshape(tensor=sample['image'], shape=self.args['input_shape'])

            if pick_dataset == 'train':
                sample['image'] = image_augm(sample['image'])
                sample['image'] = tf.image.stateless_random_flip_up_down(image=sample['image'], seed=self.seeds[:, 0])
                sample['image'] = tf.image.stateless_random_flip_left_right(image=sample['image'], seed=self.seeds[:, 1])
                sample['image'] = tf.image.stateless_random_brightness(image=sample['image'], max_delta=0.1, seed=self.seeds[:, 2])
                sample['image'] = tf.image.stateless_random_contrast(image=sample['image'], lower=0.8, upper=1.2, seed=self.seeds[:, 3])
                sample['image'] = tf.image.stateless_random_saturation(image=sample['image'], lower=0.8, upper=1.2, seed=self.seeds[:, 4])
                sample['image'] = tfa.image.sharpness(image=tf.cast(sample['image'], dtype=tf.float32), factor=self.TF_RNG.uniform(shape=[1], maxval=2., dtype=tf.float32), name='Sharpness')
                sample['image'] = tfa.image.translate(images=sample['image'], translations=self.TF_RNG.uniform(shape=[2], minval=-self.args['image_size'] * 0.05, maxval=self.args['image_size'] * 0.05, dtype=tf.float32), name='Translation')
                sample['image'] = tfa.image.rotate(images=sample['image'], angles=tf.cast(self.TF_RNG.uniform(shape=[1], minval=0, maxval=360, dtype=tf.int32), dtype=tf.float32), name='Rotation')
                sample['image'] = tf.cond(tf.math.less_equal(self.TF_RNG.uniform(shape=[1], maxval=1., dtype=tf.float32), 0.5),
                                          lambda: tfa.image.gaussian_filter2d(image=sample['image'], sigma=float(NP_RNG.random(size=1)) * 2, filter_shape=5, name='Gaussian_filter'),
                                          lambda: sample['image'])

                sample['image'] = {'xept': xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                                   'effnet0': efficientnet.preprocess_input,
                                   'effnet1': efficientnet.preprocess_input}[self.args['pretrained']](sample['image'])

            if pick_dataset == 'isic20_test':
                image_path = sample.pop('image_path')
                return sample, image_path
            else:
                return sample, label, sample_weight

        def image_augm(image):
            return image

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(tf_imread, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset.repeat(repeat).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
