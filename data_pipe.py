import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import xception, inception_v3, efficientnet

from config import DATA_MAP, BEN_MAL_MAP, NEV_MEL_MAP


class MelData:
    def __init__(self, args: dict):
        self.args = args
        self.TF_RNG = tf.random.Generator.from_seed(1312)  # .from_non_deterministic_state()
        self.seeds = self.TF_RNG.make_seeds(5)
        self.batch_size = self.args['batch_size'] * self.args['replicas']
        self.data_df = {}
        for key in ('train', 'val', 'test', 'isic20_test'):
            self.data_df[key] = self.prep_df(pd.read_csv(self.args['dir_dict']['data_csv'][key]), mode=key)
            if key == 'train':
                self.data_df[key] = self.data_df[key].sample(frac=self.args['dataset_frac'], random_state=1312)

    def prep_df(self, df: pd.DataFrame, mode: str):
        df['image'] = df['image'].apply(lambda x: os.path.join(self.args['dir_dict']['image_folder'], x))
        if self.args['task'] == 'ben_mal':
            mapper = BEN_MAL_MAP
        elif self.args['task'] == 'nev_mel':
            mapper = NEV_MEL_MAP
        else:
            mapper = None
        df.replace(to_replace=mapper, inplace=True)
        if self.args['image_type'] != 'both':  # Keep derm or clinic, samples.
            df.drop(df[df['image_type'] != DATA_MAP['image_type'][self.args['image_type']]].index, errors='ignore', inplace=True)
        if mode != 'isic20_test':
            if self.args['task'] == 'nev_mel':  # Drop : NNV, NMC, SUS, unknown
                df.drop(df[df['class'] == 2].index, errors='ignore', inplace=True)
            if self.args['task'] == '5cls':  # Drop: unknown
                df.drop(df[df['class'] == 5].index, errors='ignore', inplace=True)
        return df

    def weights(self):
        class_counts = dict(self.data_df['train']['class'].value_counts())
        image_type_counts = dict(self.data_df['train']['image_type'].value_counts())
        weights_per_image_type = len(self.data_df['train']) / (len(image_type_counts) * np.asarray([image_type_counts[k] for k in sorted(image_type_counts)]))
        weights_per_image_type = np.sqrt(weights_per_image_type)  # Through sqrt
        weights_per_class = len(self.data_df['train']) / (self.args['num_classes'] * np.asarray([class_counts[k] for k in sorted(class_counts)]))
        return weights_per_class, weights_per_image_type, image_type_counts, class_counts

    def set_sample_weights_to_df(self, df):
        weights_per_class, weights_per_image_type, image_type_counts, class_counts = self.weights()
        if self.args['task'] == '5cls':
            weights_per_class *= [1.5, 1., 1., 1., 2.5]
        if self.args['task'] in ('ben_mal', 'nev_mel'):
            weights_per_class *= [1., 2.]
        if not self.args['no_image_weights']:
            for idx1, _class in enumerate(sorted(class_counts)):
                for idx2, _image_type in enumerate(sorted(image_type_counts)):
                    df.loc[(df['image_type'] == _image_type) & (df['class'] == _class), 'sample_weights'] = weights_per_image_type[idx2]  # + self.weights_per_class[idx1]
            df['sample_weights'] /= df['sample_weights'].min()
        else:
            df['sample_weights'] = 1.
        return df

    def make_onehot(self, df, mode):
        ohe_features = {'image_path': tf.convert_to_tensor(df['image'])}
        if not self.args['only_image']:
            if not self.args['no_image_type']:
                ohe_features['image_type'] = tf.keras.backend.one_hot(indices=df['image_type'], num_classes=2)
            ohe_features['sex'] = tf.keras.backend.one_hot(indices=df['sex'], num_classes=2)
            ohe_features['age_approx'] = tf.keras.backend.one_hot(indices=df['age_approx'], num_classes=10)
            ohe_features['location'] = tf.keras.backend.one_hot(indices=df['location'], num_classes=6)
        labels = None
        sample_weights = None
        if mode != 'isic20_test':
            labels = {'class': tf.keras.backend.one_hot(indices=df['class'], num_classes=self.args['num_classes'])}
            sample_weights = df['sample_weights']
        return ohe_features, labels, sample_weights

    def get_dataset(self, mode=None, repeat=1):
        data = self.data_df[mode]
        if mode != 'isic20_test':
            data = self.set_sample_weights_to_df(data)
        data = self.make_onehot(df=data, mode=mode)

        def prep_input(sample, label=None, sample_weight=None):
            sample['image'] = tf.image.decode_image(tf.io.read_file(sample['image_path']), channels=3, dtype=tf.uint8)
            sample['image'] = tf.reshape(tensor=sample['image'], shape=self.args['input_shape'])
            for key in sample.keys():
                if key not in ('image', 'image_path'):
                    sample[key] = tf.expand_dims(input=sample[key], axis=-1)
            if mode == 'train':
                sample['image'] = tf.image.stateless_random_flip_up_down(image=sample['image'], seed=self.seeds[:, 0])
                sample['image'] = tf.image.stateless_random_flip_left_right(image=sample['image'], seed=self.seeds[:, 1])
                sample['image'] = tf.image.stateless_random_brightness(image=sample['image'], max_delta=0.1, seed=self.seeds[:, 2])
                sample['image'] = tf.image.stateless_random_contrast(image=sample['image'], lower=0.8, upper=1.2, seed=self.seeds[:, 3])
                sample['image'] = tf.image.stateless_random_saturation(image=sample['image'], lower=0.8, upper=1.2, seed=self.seeds[:, 4])
                # sample['image'] = tfa.image.sharpness(image=tf.cast(sample['image'], dtype=tf.float32), factor=self.TF_RNG.uniform(shape=[1], maxval=2.), name='Sharpness')
                sample['image'] = tfa.image.translate(images=sample['image'], translations=self.TF_RNG.uniform(shape=[2], minval=-self.args['image_size'] * 0.05, maxval=self.args['image_size'] * 0.05, dtype=tf.float32), name='Translation')
                sample['image'] = tfa.image.rotate(images=sample['image'], angles=tf.cast(self.TF_RNG.uniform(shape=[1], minval=0, maxval=360, dtype=tf.int32), dtype=tf.float32), name='Rotation')
                sample['image'] = tf.cond(tf.less(self.TF_RNG.uniform(shape=[1]), 0.6),
                                          lambda: tfa.image.gaussian_filter2d(image=sample['image'], sigma=1.5, filter_shape=5, name='Gaussian_filter'),
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
