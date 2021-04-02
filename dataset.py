from abc import ABC

import pandas as pd
import tensorflow as tf
from tensorflow import data
import numpy as np
from sklearn import utils

from hyperparameters import BATCH_SIZE_RANGE, HWC_RNG
from config import CLASSES_DICT, IMAGE_TYPE_DICT, SEX_DICT, ANATOM_SITE_DICT


def setup_data():
    all_data = pd.read_csv('all_data.csv', index_col=0)
    all_data.drop('dataset_id', axis=1, inplace=True)
    all_data = all_data[['image', 'image_type', 'sex', 'anatom_site_general', 'age_approx', 'class']]
    all_data['image_type'].fillna('nan', inplace=True)
    all_data['anatom_site_general'].fillna('nan', inplace=True)
    all_data['sex'].fillna('nan', inplace=True)
    all_data['age_approx'].fillna(1000, inplace=True)
    all_data = utils.shuffle(all_data)
    training_data = all_data[:int(len(all_data) * 0.8)][:10000]
    eval_data = all_data[int(len(all_data) * 0.8):][:1000]

    named_class_weights = dict((1 / training_data['class'].value_counts() * len(training_data)) / len(CLASSES_DICT))
    class_weights = {}
    for key in CLASSES_DICT.keys():
        class_weights[CLASSES_DICT[key]] = named_class_weights[key]
    del named_class_weights
    class_weights = list(class_weights.values())
    for split in [training_data, eval_data]:
        split['image'] = split['image'].map(lambda x: tf.convert_to_tensor(x, dtype=tf.string))
        split['image_type'] = split['image_type'].map(IMAGE_TYPE_DICT).map(
            lambda x: tf.one_hot(x, depth=len(IMAGE_TYPE_DICT), name='image_type', dtype=tf.float32))
        split['sex'] = split['sex'].map(SEX_DICT).map(
            lambda x: tf.one_hot(x, depth=len(SEX_DICT), name='sex', dtype=tf.float32))
        split['anatom_site_general'] = split['anatom_site_general'].map(ANATOM_SITE_DICT).map(
            lambda x: tf.one_hot(x, depth=len(ANATOM_SITE_DICT), name='anatom_site', dtype=tf.float32))
        split['age_approx'] = split['age_approx'].map(
            lambda x: tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.float32, name='age'), axis=0))
        split['class'] = split['class'].map(CLASSES_DICT).map(
            lambda x: tf.one_hot(x, depth=len(CLASSES_DICT), name='classes', dtype=tf.float32))
    # Shuffle data
    # msk = np.random.rand(len(all_data)) < 0.8
    # training_data = all_data[msk]
    # eval_data = all_data[~msk]
    # Split data to training and evaluation segments
    return training_data, eval_data, class_weights


def _data_gen(data_set):  # cast to Tensors. image, image_type, sex, classes, anatom_site, age,
    for sample in data_set.values:
        yield tuple(sample)


class MelData:
    def __init__(self, hparams):
        self.train_eval_datasets = []
        self.train_pd, self.eval_pd, self._weights = setup_data()
        self.train_len = len(self.train_pd)
        self.eval_len = len(self.eval_pd)
        self.hparams = hparams

    def get_datasets(self):
        for data_split in [self.train_pd, self.eval_pd]:
            dataset = data.Dataset.from_generator(lambda: _data_gen(data_split),
                                                  output_signature=(tf.TensorSpec(shape=[],
                                                                                  dtype=tf.string, name='image'),
                                                                    tf.TensorSpec(shape=len(IMAGE_TYPE_DICT),
                                                                                  dtype=tf.float32, name='image_type'),
                                                                    tf.TensorSpec(shape=len(SEX_DICT),
                                                                                  dtype=tf.float32, name='sex'),
                                                                    tf.TensorSpec(shape=len(ANATOM_SITE_DICT),
                                                                                  dtype=tf.float32,
                                                                                  name=' anatom_site'),
                                                                    tf.TensorSpec(shape=[1],
                                                                                  dtype=tf.float32, name='age'),
                                                                    tf.TensorSpec(shape=len(CLASSES_DICT),
                                                                                  dtype=tf.float32, name='classes')))
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = dataset.map(lambda image, image_type, sex, anatom_site, age, classes: (
                {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(image)), self.hparams[HWC_RNG],
                                                   self.hparams[HWC_RNG]),
                 'image_type': image_type, 'sex': sex, 'anatom_site': anatom_site, 'age': age},
                {'classes': classes}), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.hparams[BATCH_SIZE_RANGE]).cache()
            self.train_eval_datasets.append(dataset)
        return self.train_eval_datasets

    def get_weights(self):
        return self._weights
