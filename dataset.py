import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data
from tensorflow.python.data.experimental import AutoShardPolicy
from config import CLASSES_DICT, ANATOM_SITE_DICT, SEX_DICT, IMAGE_TYPE_DICT


def pd_to_np_dict(df):
    data_dct = {'image': np.asarray(df.loc[:, 'image'].values),
                'image_type': np.concatenate([np.asarray(df.loc[:, f'image_type_{key}'].values, dtype=float).reshape(-1, 1) for key in IMAGE_TYPE_DICT.keys()], axis=1),
                'sex': np.concatenate([np.asarray(df.loc[:, f'sex_{key}'].values, dtype=float).reshape(-1, 1) for key in SEX_DICT.keys()], axis=1),
                'anatom_site': np.concatenate([np.asarray(df.loc[:, f'anatom_site_general_{key}'].values, dtype=float).reshape(-1, 1) for key in ANATOM_SITE_DICT.keys()], axis=1),
                'age': np.expand_dims(np.asarray(df.loc[:, 'age_approx'].values, dtype=float), axis=1),
                'classes': np.concatenate([np.asarray(df.loc[:, f'class_{key}'].values, dtype=float).reshape(-1, 1) for key in CLASSES_DICT.keys()], axis=1)}
    del df
    return data_dct


class MelData:
    def __init__(self, size=-1, hwc=None, batch_size=None):
        all_data = pd.read_csv('all_data_v2.csv', index_col=0).sample(frac=1, random_state=1)[:size]
        self.train_data = pd_to_np_dict(all_data[:int(len(all_data) * 0.8)])
        self.eval_data = pd_to_np_dict(all_data[int(len(all_data) * 0.8):])
        self.train_len = self.train_data['classes'].shape[0]
        self.eval_len = self.eval_data['classes'].shape[0]
        self.batch_size = batch_size
        self.hw = {'target_height': hwc, 'target_width': hwc}

    def _to_dict(self, dataset):
        return {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(dataset['image'])), **self.hw),
                'image_type': dataset['image_type'], 'sex': dataset['sex'], 'anatom_site': dataset['anatom_site'],
                'age': dataset['age']}, {'classes': dataset['classes']}

    def get_class_weights(self):
        return dict(enumerate(np.divide(self.train_data['classes'].shape[0],
                                        np.multiply(np.sum(self.train_data['classes'], axis=0),
                                                    self.train_data['classes'].shape[1]))))

    def get_dataset(self, data_split='train', repeat=None):
        if data_split == 'train':
            dataset = data.Dataset.from_tensor_slices(self.train_data)
        elif data_split == 'eval':
            dataset = data.Dataset.from_tensor_slices(self.eval_data)
        else:
            raise ValueError(f"{data_split} is not a valid option. Choose between 'train' and 'eval'.")
        dataset = dataset.map(self._to_dict, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.repeat(repeat)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        return dataset.with_options(options)
