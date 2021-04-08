import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data

from config import CLASSES_DICT, ANATOM_SITE_DICT, SEX_DICT, IMAGE_TYPE_DICT


def init_datasets():
    # zipped = list(zip(image, image_type, sex, anatom_site, age, classes))
    # np.random.seed(1)
    # np.random.shuffle(zipped)
    # train_data = np.asarray(zipped[:int(len(all_data) * 0.8)])
    # images = train_data[:, 0]
    # print(images)
    # eval_data = np.asarray(zipped[int(len(all_data) * 0.8):])
    return


def get_class_weights(train_data):
    return np.divide(train_data[-1].shape[-1],
                     np.multiply(np.sum(train_data[-1], axis=0), train_data[-1].shape[0]))


def prepare_tensor_slices(data_split):
    image = data_split['image'].values
    image_type = np.concatenate([data_split[f'image_type_{key}'].values.reshape(-1, 1) for key in IMAGE_TYPE_DICT.keys()], axis=1)
    sex = np.concatenate([np.asarray(data_split[f'sex_{key}'].values, dtype=float).reshape(-1, 1) for key in SEX_DICT.keys()], axis=1)
    anatom_site = np.concatenate([np.asarray(data_split[f'anatom_site_general_{key}'].values, dtype=float).reshape(-1, 1) for key in ANATOM_SITE_DICT.keys()], axis=1)
    age = np.expand_dims(np.asarray(data_split['age_approx'].values, dtype=float), axis=1)
    classes = np.concatenate([np.asarray(data_split[f'class_{key}'].values, dtype=float).reshape(-1, 1) for key in CLASSES_DICT.keys()], axis=1)
    return image, image_type, sex, anatom_site, age, classes


class MelData:
    def __init__(self, hwc=None, batch_size=None):
        all_data = pd.read_csv('all_data_v2.csv', index_col=0).sample(frac=1, random_state=1)
        self.train_data = prepare_tensor_slices(all_data[:int(len(all_data) * 0.8)])
        self.eval_data = prepare_tensor_slices(all_data[int(len(all_data) * 0.8):])
        self.train_len = len(self.train_data)
        self.eval_len = len(self.eval_data)
        self.batch_size = batch_size
        self.hwc = hwc

    def get_dataset(self, data_split='train', repeat=None):
        if data_split == 'train':
            dataset = data.Dataset.from_tensor_slices(self.train_data)
            print(dataset.as_numpy_iterator().next())
        elif data_split == 'eval':
            dataset = data.Dataset.from_tensor_slices(self.eval_data)
        else:
            raise ValueError(f"{data_split} is not a valid option. Choose between 'train' and 'eval'.")
        # dataset = data.Dataset.from_generator(lambda: (tuple(sample) for sample in data_split),
        #                                       output_types=(
        #                                           tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
        #                                           tf.float32),
        #                                       output_shapes=(tf.TensorShape(self.train_data[:, 0][0].shape),
        #                                                      tf.TensorShape(self.train_data[:, 1][0].shape),
        #                                                      tf.TensorShape(self.train_data[:, 2][0].shape),
        #                                                      tf.TensorShape(self.train_data[:, 3][0].shape),
        #                                                      tf.TensorShape(self.train_data[:, 4][0].shape),
        #                                                      tf.TensorShape(self.train_data[:, 5][0].shape)))
        dataset = dataset.map(lambda image, image_type, sex, anatom_site, age, classes: (
            {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(image)), self.hwc, self.hwc),  # tf.convert_to_tensor(x, dtype=tf.string)
             'image_type': image_type, 'sex': sex, 'anatom_site': anatom_site, 'age': age},
            {'classes': classes}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(repeat)
        return dataset
