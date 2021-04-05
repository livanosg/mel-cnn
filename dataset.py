import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data

from config import CLASSES_DICT, ANATOM_SITE_DICT, SEX_DICT, IMAGE_TYPE_DICT


def init_datasets():
    all_data = pd.read_csv('all_data_v2.csv', index_col=0)
    image = all_data['image'].map(lambda x: tf.convert_to_tensor(x, dtype=tf.string)).values
    image_type = np.concatenate([all_data[f'image_type_{key}'].values.reshape(-1, 1) for key in IMAGE_TYPE_DICT.keys()], axis=1)
    sex = np.concatenate([all_data[f'sex_{key}'].values.reshape(-1, 1) for key in SEX_DICT.keys()], axis=1)
    anatom_site = np.concatenate([all_data[f'anatom_site_general_{key}'].values.reshape(-1, 1) for key in ANATOM_SITE_DICT.keys()], axis=1)
    age = np.expand_dims(np.asarray(all_data['age_approx'].values, dtype=float), axis=1)
    classes = np.concatenate([all_data[f'class_{key}'].values.reshape(-1, 1) for key in CLASSES_DICT.keys()], axis=1)
    zipped = list(zip(image, image_type, sex, anatom_site, age, classes))
    np.random.shuffle(zipped)
    train_data = np.asarray(zipped[:int(len(all_data) * 0.8)], dtype=object)[:500]
    eval_data = np.asarray(zipped[int(len(all_data) * 0.8):], dtype=object)
    return train_data, eval_data


def get_class_weights(train_data):
    return np.divide(train_data[:, -1].shape[-1],
                     np.multiply(np.sum(train_data[:, -1], axis=0), train_data[0, -1].shape[0]))


class MelData:
    def __init__(self, hwc=None, batch_size=None):
        self.train_data, self.eval_data = init_datasets()
        self.train_len = len(self.train_data)
        self.eval_len = len(self.eval_data)
        self.batch_size = batch_size
        self.hwc = hwc
        self.train_eval_datasets = []

    def get_dataset(self, data_split='train', repeat=None):
        if data_split == 'train':
            data_split = self.train_data
        elif data_split == 'eval':
            data_split = self.eval_data
        else:
            raise ValueError(f"{data_split} is not a valid option. Choose between 'train' and 'eval'.")
        dataset = data.Dataset.from_generator(lambda: (tuple(sample) for sample in data_split),
                                              output_signature=(tf.TensorSpec(shape=[],
                                                                              dtype=tf.string, name='image'),
                                                                tf.TensorSpec(shape=self.train_data[:, 1][0].shape,
                                                                              dtype=tf.float32, name='image_type'),
                                                                tf.TensorSpec(shape=self.train_data[:, 2][0].shape,
                                                                              dtype=tf.float32, name='sex'),
                                                                tf.TensorSpec(shape=self.train_data[:, 3][0].shape,
                                                                              dtype=tf.float32,
                                                                              name=' anatom_site'),
                                                                tf.TensorSpec(shape=[1],
                                                                              dtype=tf.float32, name='age'),
                                                                tf.TensorSpec(shape=self.train_data[:, 5][0].shape,
                                                                              dtype=tf.float32, name='classes')))
        dataset = dataset.map(lambda image, image_type, sex, anatom_site, age, classes: (
            {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(image)), self.hwc, self.hwc),
             'image_type': image_type, 'sex': sex, 'anatom_site': anatom_site, 'age': age},
            {'classes': classes}), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat(repeat)
        return dataset
