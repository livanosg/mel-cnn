import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import data
from sklearn import utils
from tensorflow.python.data.experimental import AutoShardPolicy

import config

all_data = pd.read_csv('all_data.csv', index_col=0)
all_data.drop('dataset_id', axis=1, inplace=True)
all_data['image_type'].fillna('nan', inplace=True)
all_data['anatom_site_general'].fillna('nan', inplace=True)
all_data['sex'].fillna('nan', inplace=True)
all_data = utils.shuffle(all_data)  # Shuffle data

# Split data to training and evaluation segments
msk = np.random.rand(len(all_data)) < 0.8
training_data = all_data[msk]
eval_data = all_data[~msk]

anatom_site_dict = {'head neck': 0, 'torso': 1, 'lateral torso': 2, 'upper extremity': 3, 'lower extremity': 4,
                    'palms soles': 5, 'oral genital': 6, 'nan': 7}
image_type_dict = {'clinic': 0, 'derm': 1}
sex_dict = {'m': 0, 'f': 1, 'nan': 2}
classes_dict = {'NV': 0, "NNV": 1, "MEL": 2, "NMC": 3, "SUS": 4}


class MelData(tf.data.Dataset):

    @staticmethod
    def _data_gen(mode):  # cast to Tensors. image, image_type, sex, classes, anatom_site, age,
        mode = mode.decode("utf-8")
        if mode == 'train':
            mode_data = training_data
        elif mode == 'eval':
            mode_data = eval_data
        else:
            raise ValueError(f'Unknown mode {mode}.')
        for data in mode_data.values:
            image = tf.convert_to_tensor(data[0])
            image_type = tf.one_hot(image_type_dict[data[1]], depth=len(image_type_dict), name='image_type')
            sex = tf.one_hot(sex_dict[data[2]], depth=len(sex_dict), name='sex')
            anatom_site = tf.one_hot(anatom_site_dict[data[4]], depth=len(anatom_site_dict), name='anatom_site')
            age = tf.expand_dims(tf.convert_to_tensor(data[5], dtype=tf.float32, name='age'), axis=0)
            classes = tf.one_hot(classes_dict[data[3]], depth=len(classes_dict), name='classes')
            yield image, image_type, sex, anatom_site, age, classes

    @staticmethod
    def _read_image(data, hwc=config.HWC):
        data['image'] = tf.io.read_file(filename=data['image'])
        data['image'] = tf.image.decode_image(contents=data['image'], channels=3, dtype=tf.float32)
        data['image'] = tf.image.resize_with_pad(data['image'], hwc[0], hwc[1])
        return data

    def __new__(cls, mode='train', *args, **kwargs):
        train_eval_datasets = []
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        for mode in ('train', 'eval'):
            dataset = data.Dataset.from_generator(cls._data_gen,
                                                  output_signature=(tf.TensorSpec(shape=[],  # 'image':
                                                                                  dtype=tf.string, name='image'),
                                                                    tf.TensorSpec(shape=len(image_type_dict),
                                                                                  # 'image_type':
                                                                                  dtype=tf.float32, name='image_type'),
                                                                    tf.TensorSpec(shape=len(sex_dict),  # 'sex':
                                                                                  dtype=tf.float32, name='sex'),
                                                                    tf.TensorSpec(shape=len(anatom_site_dict),
                                                                                  # 'anatom_site':
                                                                                  dtype=tf.float32, name='anatom_site'),
                                                                    tf.TensorSpec(shape=[1],  # 'age':
                                                                                  dtype=tf.float32, name='age'),
                                                                    tf.TensorSpec(shape=len(classes_dict),  # 'classes':
                                                                                  dtype=tf.float32, name='classes')), args=[mode])
            dataset = dataset.map(lambda image, image_type, sex, anatom_site, age, classes: {'image': image,
                                                                                             'image_type': image_type,
                                                                                             'sex': sex,
                                                                                             'anatom_site': anatom_site,
                                                                                             'age': age,
                                                                                             'classes': classes})
            dataset = dataset.prefetch(config.BUFFER_SIZE)
            dataset = dataset.map(lambda data: cls._read_image(data), num_parallel_calls=tf.data.AUTOTUNE).cache()
            dataset = dataset.map(lambda data: ({'image': data['image'], 'image_type': data['image_type'],
                                                 'sex': data['sex'], 'anatom_site': data['anatom_site'],
                                                 'age': data['age']}, {'classes': data['classes']})).cache()
            dataset = dataset.batch(config.BATCH_SIZE)
            # dataset = dataset.repeat()
            dataset = dataset.with_options(options)
            train_eval_datasets.append(dataset)
        return train_eval_datasets


if __name__ == '__main__':
    _, _1 = MelData()
    print(_)
    print(_1)
    test = _1.as_numpy_iterator().next()[0]['image'][0]
    print(test)
    #
    #
    # for i in test:
    #     print(i)
