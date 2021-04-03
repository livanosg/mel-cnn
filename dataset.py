import pandas as pd
import tensorflow as tf
from tensorflow import data
import numpy as np
from hyperparameters import BATCH_SIZE_RANGE, HWC_RNG


def setup_data():
    all_data = pd.read_csv('all_data_v2.csv', index_col=0)
    image = all_data['image'].map(lambda x: tf.convert_to_tensor(x, dtype=tf.string)).values
    image_type = np.concatenate((all_data['image_type_derm'].values.reshape(-1, 1),
                                 all_data['image_type_clinic'].values.reshape(-1, 1)),
                                axis=1)
    sex = np.concatenate((all_data['sex_m'].values.reshape(-1, 1),
                          all_data['sex_f'].values.reshape(-1, 1)),
                         axis=1)
    anatom_site = np.concatenate((all_data['anatom_site_general_head/neck'].values.reshape(-1, 1),
                                  all_data['anatom_site_general_torso'].values.reshape(-1, 1),
                                  all_data['anatom_site_general_lateral_torso'].values.reshape(-1, 1),
                                  all_data['anatom_site_general_lower_extremity'].values.reshape(-1, 1),
                                  all_data['anatom_site_general_oral/genital'].values.reshape(-1, 1),
                                  all_data['anatom_site_general_palms/soles'].values.reshape(-1, 1),
                                  ), axis=1)
    age = np.expand_dims(np.asarray(all_data['age_approx'].values, dtype=float), axis=1)
    classes = np.concatenate((all_data['class_MEL'].values.reshape(-1, 1),
                              all_data['class_NMC'].values.reshape(-1, 1),
                              all_data['class_NNV'].values.reshape(-1, 1),
                              all_data['class_NV'].values.reshape(-1, 1),
                              all_data['class_SUS'].values.reshape(-1, 1)), axis=1)

    zipped = list(zip(image, image_type, sex, anatom_site, age, classes))
    np.random.shuffle(zipped)
    train_data = np.asarray(zipped[:int(len(all_data) * 0.8)], dtype=object)[:1000]
    eval_data = np.asarray(zipped[int(len(all_data) * 0.8):], dtype=object)[:500]
    return train_data, eval_data


def get_class_weights(train_data):
    return np.divide(train_data[:, -1].shape[-1], np.multiply(np.sum(train_data[:, -1], axis=0), train_data[0, -1].shape[0]))


def _data_gen(data_set):  # cast to Tensors. image, image_type, sex, classes, anatom_site, age,
    for sample in data_set:
        yield tuple(sample)


class MelData:
    def __init__(self, hparams):
        self.train_eval_datasets = []
        self.train_data, self.eval_data = setup_data()
        print(self.train_data[:, :][0].shape)
        self.train_len = len(self.train_data)
        self.eval_len = len(self.eval_data)
        self.hparams = hparams

    def get_datasets(self):
        for data_split in [self.train_data, self.eval_data]:
            dataset = data.Dataset.from_generator(lambda: _data_gen(data_split),
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
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = dataset.map(lambda image, image_type, sex, anatom_site, age, classes: (
                {'image': tf.image.resize_with_pad(tf.image.decode_image(tf.io.read_file(image)), self.hparams[HWC_RNG],
                                                   self.hparams[HWC_RNG]),
                 'image_type': image_type, 'sex': sex, 'anatom_site': anatom_site, 'age': age},
                {'classes': classes}), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.hparams[BATCH_SIZE_RANGE]).cache()
            self.train_eval_datasets.append(dataset)
        return self.train_eval_datasets


if __name__ == '__main__':
    train, eval = setup_data()
    print(np.vstack(train[:, -1]).shape)

    # b = _data_gen(a)
    # b = np.asarray(list(b))
    #
    # weights = get_class_weights(b)
    # print(weights)
    # print('-----------========================-------------------')
    # print(np.sum(b[:, -1]))
    # for i in b:
    #     print (i)

    # print(weights)
