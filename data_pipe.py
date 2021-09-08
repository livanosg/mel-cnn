import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from config import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES, MAIN_DIR, ISIC20_TEST_PATH, \
    TEST_CSV_PATH, VAL_CSV_PATH, TRAIN_CSV_PATH
from sklearn.preprocessing import OneHotEncoder


class MelData:
    def __init__(self, image_type: str, task: str, dir_dict: dict, dataset_frac: float, pretrained: str, input_shape: list):
        self.image_type = image_type
        self.task = task
        self.pretrained = pretrained
        self.input_shape = input_shape
        self.TF_RNG = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.class_names = TASK_CLASSES[self.task]
        self.num_classes = len(self.class_names)
        self.dir_dict = dir_dict
        if self.image_type != 'both':
            self.image_types = [self.image_type]
        else:
            self.image_types = IMAGE_TYPE
        self.data_df = {'train': self.prep_df(mode='train').sample(frac=dataset_frac),
                        'validation': self.prep_df(mode='validation').sample(frac=dataset_frac),
                        'test': self.prep_df(mode='test'),
                        'isic20_test': self.prep_df(mode='isic20_test')}
        self.train_len = len(self.data_df['train'])

    def prep_df(self, mode: str):
        df = pd.read_csv({'train': TRAIN_CSV_PATH, 'validation': VAL_CSV_PATH, 'test': TEST_CSV_PATH, 'isic20_test': ISIC20_TEST_PATH}[mode])
        df['image'] = df['image'].apply(lambda x: os.path.join(self.dir_dict['image_folder'], x))
        if mode != 'isic20_test':
            if self.task == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            elif self.task in ('nev_mel', '5cls'):
                if self.task == 'nev_mel':
                    df.drop(df[df['class'].isin(['NNV', 'SUS', 'NMC'])].index, errors='ignore', inplace=True)
                df.drop(df[df['class'] == 'UNK'].index, errors='ignore', inplace=True)
            if self.image_type != 'both':  # Keep derm or clinic, samples.
                df.drop(df[df['image_type'] != self.image_type].index, errors='ignore', inplace=True)
        return df

    def logs(self):
        for key in ('train', 'validation', 'test'):
            df = self.data_df[key]
            img_type_dict = {}
            for img_tp in self.image_types:
                cat_dict = {}
                for cat in ['sex', 'age_approx', 'location', 'image_type']:
                    class_dict = {}
                    for class_ in self.class_names:
                        counts = df.loc[(df['image_type'] == img_tp) & (df['class'] == class_), cat].value_counts(dropna=False)
                        if np.nan in counts.keys():
                            keys = sorted(counts.keys().drop(np.nan)) + [np.nan]
                        else:
                            keys = sorted(counts.keys())
                        counts = {key: counts[key] for key in keys}
                        class_dict[class_] = dict(counts)
                    cat_dict[cat] = pd.DataFrame.from_dict(class_dict)
                img_type_dict[img_tp] = pd.concat(cat_dict)
            new_df = pd.concat(img_type_dict, keys=list(img_type_dict.keys()))
            os.makedirs(os.path.join(MAIN_DIR, 'data_info'), exist_ok=True)
            new_df.to_csv(os.path.join(MAIN_DIR, 'data_info', 'descr_{}_{}_{}.csv'.format(self.task, self.image_type, key)))

    def oversampling(self, df):
        for _image_type in self.image_types:
            class_weights = len(df) / df.loc[df['image_type'] == _image_type, 'class'].value_counts()
            for _class in self.class_names:
                df.loc[(df['image_type'] == _image_type) & (df['class'] == _class), 'sample_weights'] = class_weights[_class]
        df = df.sample(frac=5., replace=True, weights='sample_weights')
        return df

    def make_onehot(self, df, mode, no_image_type, only_image):
        ohe_features = {'image_path': df['image']}
        if not only_image:
            categories = [LOCATIONS, SEX, AGE_APPROX]
            columns = ['location', 'sex', 'age_approx']
            if not no_image_type:
                categories.append(IMAGE_TYPE)
                columns.append('image_type')
            features_env = OneHotEncoder(handle_unknown='ignore', categories=categories).fit(self.data_df['train'][columns])
            ohe_features['clinical_data'] = features_env.transform(df[columns]).toarray()
        if mode == 'isic20_test':
            labels = ohe_features['image_path']
        else:
            label_enc = OneHotEncoder(categories=[self.class_names])
            label_enc.fit(self.data_df['train']['class'].values.reshape(-1, 1))
            labels = {'class': label_enc.transform(df['class'].values.reshape(-1, 1)).toarray()}
        return ohe_features, labels

    def get_dataset(self, mode=None, repeat=1, batch=16, no_image_type=False, only_image=False):
        data = self.data_df[mode]
        if mode == 'train':
            data = self.oversampling(data)
        data = self.make_onehot(df=data, mode=mode, no_image_type=no_image_type, only_image=only_image)
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if mode == 'train':
            dataset = dataset.shuffle(buffer_size=len(self.data_df['train']), reshuffle_each_iteration=True)
            dataset = dataset.batch(batch, drop_remainder=True)
        else:
            dataset = dataset.batch(batch)
        dataset = dataset.map(lambda sample, label: self.prep_input(sample=sample, label=label, mode=mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode == 'train':
            dataset = dataset.map(lambda sample, label: self.augm(sample, label, batch), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        if mode == 'train':
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(1)
        return dataset.prefetch(buffer_size=batch)

    def all_datasets(self, repeat=1, batch=16, no_image_type=False, only_image=False):
        return {'train': self.get_dataset('train', repeat=repeat, batch=batch, no_image_type=no_image_type, only_image=only_image),
                'validation': self.get_dataset('validation', repeat=repeat, batch=batch, no_image_type=no_image_type, only_image=only_image),
                'test': self.get_dataset('test', repeat=repeat, batch=batch, no_image_type=no_image_type, only_image=only_image),
                'isic20_test': self.get_dataset('isic20_test', repeat=repeat, batch=batch, no_image_type=no_image_type, only_image=only_image)}

    @tf.function()
    def prep_input(self, sample, label, mode):
        modified = {'image': tf.map_fn(lambda x: tf.reshape(tensor=tf.cast(tf.image.decode_image(tf.io.read_file(x), channels=3, dtype=tf.uint8),
                                                                           dtype=tf.float32), shape=self.input_shape),
                                       elems=sample['image_path'],
                                       fn_output_signature=tf.float32),
                    'image_path': sample['image_path'],
                    'clinical_data': sample['clinical_data']}
        if mode == 'isic20_test':
            label = modified.pop('image_path')
        else:
            modified.pop('image_path')
        return modified, label

    def augm(self, sample, label, batch):
        sample['image'] = tf.image.random_flip_up_down(image=sample['image'])
        sample['image'] = tf.image.random_flip_left_right(image=sample['image'])
        sample['image'] = tf.image.random_brightness(image=sample['image'], max_delta=60.)
        sample['image'] = tf.image.random_contrast(image=sample['image'], lower=.5, upper=1.5)
        sample['image'] = tf.clip_by_value(sample['image'], clip_value_min=0., clip_value_max=255.)
        sample['image'] = tf.image.random_saturation(image=sample['image'], lower=0.8, upper=1.2)
        # sample['image'] = tfa.image.sharpness(image=tf.cast(sample['image'], dtype=tf.float32), factor=self.TF_RNG.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness')
        trans_val = self.input_shape[0] * 0.2
        sample['image'] = tfa.image.translate(images=sample['image'], translations=self.TF_RNG.uniform(shape=[2], minval=-trans_val, maxval=trans_val, dtype=tf.float32), name='Translation')
        sample['image'] = tfa.image.rotate(images=sample['image'], angles=tf.cast(
            self.TF_RNG.uniform(shape=[batch], minval=0, maxval=360, dtype=tf.int32), dtype=tf.float32),
                                           interpolation='bilinear', name='Rotation')
        sample['image'] = tf.cond(tf.less(self.TF_RNG.uniform(shape=[1]), 0.5),
                                  lambda: tfa.image.gaussian_filter2d(image=sample['image'], sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                                  lambda: sample['image'])
        cutout_ratio = 0.15
        for i in range(3):
            mask_height = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio), dtype=tf.int32) * 2
            mask_width = tf.cast(self.TF_RNG.uniform(shape=[], minval=0, maxval=self.input_shape[0] * cutout_ratio), dtype=tf.int32) * 2
            sample['image'] = tfa.image.random_cutout(sample['image'], mask_size=(mask_height, mask_width))
        sample['image'] = {'xept': xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                           'effnet0': efficientnet.preprocess_input, 'effnet1': efficientnet.preprocess_input,
                           'effnet6': efficientnet.preprocess_input}[self.pretrained](sample['image'])
        return sample, label
