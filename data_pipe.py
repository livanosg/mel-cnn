import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from config import BEN_MAL_MAP, LOCATIONS, IMAGE_TYPE, SEX, AGE_APPROX, TASK_CLASSES
from sklearn.preprocessing import OneHotEncoder


class MelData:
    def __init__(self, args: dict):
        self.args = args
        self.TF_RNG = tf.random.Generator.from_non_deterministic_state()  # .from_seed(1312)
        self.batch_size = self.args['batch_size'] * self.args['replicas']
        self.data_df = {}
        self.classes = TASK_CLASSES[self.args['task']]
        if self.args['image_type'] != 'both':
            self.image_types = [self.args['image_type']]
        else:
            self.image_types = IMAGE_TYPE
        for key in ('train', 'val', 'test', 'isic20_test'):
            self.data_df[key] = self.prep_df(pd.read_csv(self.args['dir_dict']['data_csv'][key]), mode=key)
            if key in ('train', 'val'):
                self.data_df[key] = self.data_df[key].sample(frac=self.args['dataset_frac'], random_state=1312)

    def prep_df(self, df: pd.DataFrame, mode: str):
        df['image'] = df['image'].apply(lambda x: os.path.join(self.args['dir_dict']['image_folder'], x))
        if mode != 'isic20_test':
            if self.args['task'] == 'ben_mal':
                df.replace(to_replace=BEN_MAL_MAP, inplace=True)
            else:
                if self.args['task'] == 'nev_mel':
                    df.drop(df[df['class'].isin(['NNV', 'SUS', 'NMC'])].index, errors='ignore', inplace=True)
                df.drop(df[df['class'] == 'UNK'].index, errors='ignore', inplace=True)

        if self.args['image_type'] != 'both':  # Keep derm or clinic, samples.
            df.drop(df[df['image_type'] != self.args['image_type']].index, errors='ignore', inplace=True)
        return df

    def weights(self):
        class_counts = dict(self.data_df['train']['class'].value_counts())
        image_type_counts = dict(self.data_df['train']['image_type'].value_counts())

        weights_per_image_type = {k: np.sqrt(len(self.data_df['train']) / (len(image_type_counts) * image_type_counts[k])) for k in self.image_types}
        weights_per_class = {k: len(self.data_df['train']) / (self.args['num_classes'] * class_counts[k]) for k in self.classes}
        if self.args['task'] == '5cls':
            class_multipl = {key: [1.5, 1., 1., 1., 2.5][idx] for idx, key in enumerate(self.classes)}
        else:
            class_multipl = {key: [1., 2.5][idx] for idx, key in enumerate(self.classes)}
        for key in self.classes:
            weights_per_class[key] = weights_per_class[key] * class_multipl[key]
        return weights_per_class, weights_per_image_type, image_type_counts, class_counts

    def set_sample_weights_to_df(self, df):
        weights_per_class, weights_per_image_type, image_type_counts, class_counts = self.weights()
        if not self.args['no_image_weights']:
            for _class in self.classes:
                for _image_type in self.image_types:
                    df.loc[(df['image_type'] == _image_type) & (df['class'] == _class), 'sample_weights'] = (weights_per_image_type[_image_type] * weights_per_class[_class])
        else:
            df['sample_weights'] = 1.
        return df

    def make_onehot(self, df, mode):
        ohe_features = {'image_path': df['image']}
        if not self.args['only_image']:
            categories = [LOCATIONS, SEX, AGE_APPROX]
            columns = ['location', 'sex', 'age_approx']
            if not self.args['no_image_type']:
                categories.append(IMAGE_TYPE)
                columns.append('image_type')
            ohe_features['clinical_data'] = OneHotEncoder(handle_unknown='ignore', categories=categories).fit_transform(df[columns]).toarray()
        labels = None
        sample_weights = None
        if mode != 'isic20_test':
            labels = {'class': OneHotEncoder(categories=[self.classes]).fit_transform(df['class'].values.reshape(-1, 1)).toarray()}
            sample_weights = df['sample_weights']
        return ohe_features, labels, sample_weights

    def get_dataset(self, mode=None, repeat=1):
        data = self.data_df[mode]
        if mode != 'isic20_test':
            data = self.set_sample_weights_to_df(data)
        data = self.make_onehot(df=data, mode=mode)

        def prep_input(sample, label=None, sample_weight=None):
            sample['image'] = tf.cast(tf.image.decode_image(tf.io.read_file(sample['image_path']), channels=3, dtype=tf.uint8), tf.float32)
            sample['image'] = tf.reshape(tensor=sample['image'], shape=self.args['input_shape'])
            for key in sample.keys():
                if key not in ('image', 'image_path'):
                    sample[key] = tf.expand_dims(input=sample[key], axis=-1)
            if mode == 'train':
                sample['image'] = tf.image.random_flip_up_down(image=sample['image'])
                sample['image'] = tf.image.random_flip_left_right(image=sample['image'])
                sample['image'] = tf.image.random_brightness(image=sample['image'], max_delta=60.)
                sample['image'] = tf.image.random_contrast(image=sample['image'], lower=.5, upper=1.5)
                sample['image'] = tf.clip_by_value(sample['image'], clip_value_min=0., clip_value_max=255.)
                sample['image'] = tf.image.random_saturation(image=sample['image'], lower=0.8, upper=1.2)
                sample['image'] = tfa.image.sharpness(image=tf.cast(sample['image'], dtype=tf.float32), factor=self.TF_RNG.uniform(shape=[1], minval=0.5, maxval=1.5), name='Sharpness')
                sample['image'] = tfa.image.translate(images=sample['image'], translations=self.TF_RNG.uniform(shape=[2], minval=-self.args['image_size'] * 0.2, maxval=self.args['image_size'] * 0.2, dtype=tf.float32), name='Translation')
                sample['image'] = tfa.image.rotate(images=sample['image'], angles=tf.cast(self.TF_RNG.uniform(shape=[1], minval=0, maxval=360, dtype=tf.int32), dtype=tf.float32), interpolation='bilinear', name='Rotation')
                sample['image'] = tf.cond(tf.less(self.TF_RNG.uniform(shape=[1]), 0.5),
                                          lambda: tfa.image.gaussian_filter2d(image=sample['image'], sigma=1.5, filter_shape=3, name='Gaussian_filter'),
                                          lambda: sample['image'])
                sample['image'] = {'xept': xception.preprocess_input, 'incept': inception_v3.preprocess_input,
                                   'effnet0': efficientnet.preprocess_input,
                                   'effnet1': efficientnet.preprocess_input}[self.args['pretrained']](sample['image'])
            if mode == 'isic20_test':
                image_path = sample.pop('image_path')
                return sample, image_path
            else:
                return sample, label, sample_weight

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(prep_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset.repeat(repeat).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
