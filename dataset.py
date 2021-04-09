import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data

from config import CLASSES_DICT, ANATOM_SITE_DICT, SEX_DICT, IMAGE_TYPE_DICT


def get_class_weights(train_data):
    return np.divide(train_data['classes'].shape[0],
                     np.multiply(np.sum(train_data['classes'], axis=0), train_data['classes'].shape[1]))


def data_dict(df):
    data_dct = {'image': np.asarray(df['image'].values), 'image_type': np.concatenate(
        [np.asarray(df[f'image_type_{key}'].values, dtype=float).reshape(-1, 1) for key in IMAGE_TYPE_DICT.keys()],
        axis=1), 'sex': np.concatenate(
        [np.asarray(df[f'sex_{key}'].values, dtype=float).reshape(-1, 1) for key in SEX_DICT.keys()], axis=1),
                'anatom_site': np.concatenate(
                    [np.asarray(df[f'anatom_site_general_{key}'].values, dtype=float).reshape(-1, 1) for key in
                     ANATOM_SITE_DICT.keys()], axis=1),
                'age': np.expand_dims(np.asarray(df['age_approx'].values, dtype=float), axis=1),
                'classes': np.concatenate(
                    [np.asarray(df[f'class_{key}'].values, dtype=float).reshape(-1, 1) for key in CLASSES_DICT.keys()],
                    axis=1)}
    [df.drop(f'image_type_{key}', axis=1, inplace=True) for key in IMAGE_TYPE_DICT.keys()]
    [df.drop(f'sex_{key}', axis=1, inplace=True) for key in SEX_DICT.keys()]
    [df.drop(f'anatom_site_general_{key}', axis=1, inplace=True) for key in ANATOM_SITE_DICT.keys()]
    df.drop('age_approx', axis=1, inplace=True)
    [df.drop(f'class_{key}', axis=1) for key in CLASSES_DICT.keys()]
    return data_dct


class MelData:
    def __init__(self, size=-1, hwc=None, batch_size=None):
        all_data = pd.read_csv('all_data_v2.csv', index_col=0).sample(frac=1, random_state=1)[:size]
        self.train_data = data_dict(all_data[:int(len(all_data) * 0.8)])
        self.eval_data = data_dict(all_data[int(len(all_data) * 0.8):])
        self.train_len = self.train_data['classes'].shape[0]
        self.eval_len = self.eval_data['classes'].shape[0]
        self.batch_size = batch_size
        self.hwc = {'target_height': hwc, 'target_width': hwc}

    def to_dict(self, dataset):
        return {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(dataset['image'])), **self.hwc),
                'image_type': dataset['image_type'], 'sex': dataset['sex'], 'anatom_site': dataset['anatom_site'],
                'age': dataset['age']}, {'classes': dataset['classes']}

    def get_dataset(self, data_split='train', repeat=None):
        if data_split == 'train':
            dataset = data.Dataset.from_tensor_slices(self.train_data)
        elif data_split == 'eval':
            dataset = data.Dataset.from_tensor_slices(self.eval_data)
        else:
            raise ValueError(f"{data_split} is not a valid option. Choose between 'train' and 'eval'.")
        dataset = dataset.map(self.to_dict, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat(repeat)
        return dataset
