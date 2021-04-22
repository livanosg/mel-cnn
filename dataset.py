import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data
from config import CLASSES_DICT, ANATOM_SITE_DICT, SEX_DICT, IMAGE_TYPE_DICT


def train_val_dct(frac):
    def pd_to_dct(df):
        data_dct = {'image': np.asarray(df.loc[:, 'image']),
                    'image_type': np.concatenate([np.asarray(df.loc[:, f'image_type_{key}'], dtype=np.float16).reshape(-1, 1) for key in IMAGE_TYPE_DICT.keys()], axis=1),
                    'sex': np.concatenate([np.asarray(df.loc[:, f'sex_{key}'], dtype=np.float16).reshape(-1, 1) for key in SEX_DICT.keys()], axis=1),
                    'anatom_site': np.concatenate([np.asarray(df.loc[:, f'anatom_site_general_{key}'], dtype=np.float16).reshape(-1, 1) for key in ANATOM_SITE_DICT.keys()], axis=1),
                    'age': np.expand_dims(np.asarray(df.loc[:, 'age_approx'], dtype=np.float16), axis=1),
                    'classes': np.concatenate([np.asarray(df.loc[:, f'class_{key}'], dtype=np.float16).reshape(-1, 1) for key in CLASSES_DICT.keys()], axis=1)}
        del df
        return data_dct
    all_data = pd.read_csv('all_data_v2.csv')
    train_data = []
    val_data = []
    for col_name in ["class_MEL", "class_NMC", "class_NNV", "class_NV", "class_SUS"]:
        class_data = all_data[all_data.loc[:, col_name] == 1]
        class_train = class_data.sample(frac=0.8, random_state=1312)
        train_data.append(class_train.sample(frac=frac, random_state=1312))
        val_data.append(class_data[~class_data.index.isin(class_train.index)].sample(frac=frac, random_state=1312))
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    return pd_to_dct(train_data), pd_to_dct(val_data), len(train_data), len(val_data)


class MelData:
    def __init__(self, frac=1., hwc=None, batch_size=None):
        self.train_data, self.eval_data, self.train_len, self.eval_len = train_val_dct(frac=frac)
        self.batch_size = batch_size
        self.hw = hwc

    def _to_dict(self, sample):
        return {'image': tf.image.decode_image(tf.io.read_file("proc_" + str(self.hw) + "/" + sample['image'])),
                'image_type': sample['image_type'], 'sex': sample['sex'], 'anatom_site': sample['anatom_site'],
                'age': sample['age']}, {'classes': sample['classes']}

    def get_class_weights(self):
        return dict(enumerate(np.divide(self.train_len, np.multiply(np.count_nonzero(self.train_data['classes'], axis=0), 5))))

    def get_dataset(self, data_split='train', repeat=None):
        if data_split == 'train':
            dataset = data.Dataset.from_tensor_slices(self.train_data)
        elif data_split == 'eval':
            dataset = data.Dataset.from_tensor_slices(self.eval_data)
        else:
            raise ValueError(f"{data_split} is not a valid option. Choose between 'train' and 'eval'.")
        dataset = dataset.map(self._to_dict, num_parallel_calls=tf.data.AUTOTUNE).cache()
        dataset = dataset.batch(self.batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(tf.data.AUTOTUNE)
