import os
# import cv2
from config import MAPPER, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER, CLASS_NAMES, MAIN_DIR, directories
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, dir_dict: dict, args: dict, batch: int = 8):
        self.random_state = 0
        self.args = args
        self.batch = batch
        self.dir_dict = dir_dict
        self.image_folder = self.dir_dict["image_folder"]
        self.class_names = CLASS_NAMES[self.args['mode']]
        self.num_classes = len(self.class_names)
        self.train_data_df = pd.read_csv(self.dir_dict["data_csv"]["train"])
        self.val_data_df = pd.read_csv(self.dir_dict["data_csv"]["val"])
        self.test_data_df = pd.read_csv(self.dir_dict["data_csv"]["test"])
        if self.args['mode'] == 'ben_mal':
            self.train_data_df.replace(to_replace=BEN_MAL_MAPPER, inplace=True)
            self.val_data_df.replace(to_replace=BEN_MAL_MAPPER, inplace=True)
            self.test_data_df.replace(to_replace=BEN_MAL_MAPPER, inplace=True)
        if self.args['mode'] == 'nev_mel':
            self.train_data_df.replace(to_replace=NEV_MEL_OTHER_MAPPER, inplace=True)
            self.val_data_df.replace(to_replace=NEV_MEL_OTHER_MAPPER, inplace=True)
            self.test_data_df.replace(to_replace=NEV_MEL_OTHER_MAPPER, inplace=True)
        if self.args['image_type'] != 'both':  # Keep derm or clinic, samples.
            self.train_data_df = self.train_data_df[self.train_data_df['image_type'] == MAPPER['image_type'][self.args['image_type']]]
            self.val_data_df = self.val_data_df[self.val_data_df['image_type'] == MAPPER['image_type'][self.args['image_type']]]
            self.test_data_df = self.test_data_df[self.test_data_df['image_type'] == MAPPER['image_type'][self.args['image_type']]]
        if self.args['mode'] in ['ben_mal', 'nev_mel']:
            self.train_data_df = self.train_data_df[self.train_data_df['class'] != 2]
            self.val_data_df = self.val_data_df[self.val_data_df['class'] != 2]
            self.test_data_df = self.test_data_df[self.test_data_df['class'] != 2]
        if self.args['mode'] in ['5cls']:
            self.train_data_df = self.train_data_df[self.train_data_df['class'] != 5]
            self.val_data_df = self.val_data_df[self.val_data_df['class'] != 5]
            self.test_data_df = self.test_data_df[self.test_data_df['class'] != 5]

        self.class_counts = dict(self.train_data_df['class'].value_counts(sort=False, ascending=True))
        self.weights_per_class = np.divide(len(self.train_data_df),
                                           np.multiply(self.num_classes, [self.class_counts[k]for k in sorted(self.class_counts.keys())]))
        self.image_type_counts = dict(self.train_data_df["image_type"].value_counts(sort=False, ascending=True))
        self.weights_per_image_type = np.divide(np.sum(len(self.train_data_df)),
                                                np.multiply(len(self.image_type_counts), [self.image_type_counts[k] for k in sorted(self.image_type_counts)]))
        self.prep_train_data_df = self.prep_classes(self.train_data_df)
        self.prep_val_data_df = self.prep_classes(self.val_data_df)
        self.prep_test_data_df = self.prep_classes(self.test_data_df)
        self.train_data = self.ohe_map(dict(self.train_data_df.sample(frac=self.args["dataset_frac"])))
        self.val_data = self.ohe_map(dict(self.val_data_df.sample(frac=self.args["dataset_frac"])))
        self.test_data = self.ohe_map(dict(self.test_data_df))
        self.attr = self.dataset_attributes()
        self.data_info('all')
        self.data_info('train')
        self.data_info('val')
        self.data_info('test')

    def prep_classes(self, data):
        data["image"] = f"{self.image_folder}{os.sep}" + data["image"]
        data['image_type_weights'] = 1.
        data['class_weights'] = 1.
        for ixd, image_type in enumerate(sorted(self.image_type_counts)):
            data.loc[data['image_type'] == image_type, 'image_type_weights'] = self.weights_per_image_type[ixd]
        for idx, _class in enumerate(sorted(self.class_counts)):
            data.loc[data['class'] == _class, 'class_weights'] = self.weights_per_class[idx]
        data['sample_weights'] = data['image_type_weights']  # /2 + data['class_weights']/2  # todo check weighting formula. integrate
        return data

    def ohe_map(self, features):
        """ Turn features to one-hot encoded vectors.
        Inputs:
            features: dictionary of features int encoded.
        Outputs:
            features: dict, labels: dict
            or
            features: dict, labels: dict, sample_weights
        """
        features["image_type"] = tf.keras.backend.one_hot(indices=np.asarray(features["image_type"]), num_classes=2)
        features["sex"] = tf.keras.backend.one_hot(indices=np.asarray(features["sex"]), num_classes=2)
        features["age_approx"] = tf.keras.backend.one_hot(indices=np.asarray(features["age_approx"]), num_classes=10)
        features["anatom_site_general"] = tf.keras.backend.one_hot(indices=np.asarray(features["anatom_site_general"]), num_classes=6)
        features["class"] = tf.keras.backend.one_hot(indices=np.asarray(features["class"]), num_classes=self.num_classes)
        labels = {"class": features.pop("class")}
        sample_weights = features.pop("sample_weights")

        return features, labels, sample_weights

    def data_info(self, mode):
        dataset_info_dict = {}
        image_type_inv = {}
        if mode == 'all':
            df = self.train_data_df.append(self.val_data_df).append(self.test_data_df)
        elif mode == 'train':
            df = self.train_data_df
        elif mode == 'val':
            df = self.val_data_df
        else:
            df = self.test_data_df
        for (key, value) in MAPPER['image_type'].items():
            image_type_inv[value] = key
        for dataset_id in df['dataset_id'].unique():
            dataset_part = df[df.loc[:, 'dataset_id'] == dataset_id]  # fraction per class
            dataset_img_type_dict = {}
            for image_type in dataset_part['image_type'].unique():
                dataset_image_part = dataset_part[dataset_part.loc[:, 'image_type'] == image_type]
                class_counts = dataset_image_part['class'].value_counts()
                dataset_class_dict = {}
                for k, v in class_counts.items():
                    dataset_class_dict[CLASS_NAMES[self.args['mode']][k]] = v
                dataset_img_type_dict[image_type_inv[image_type]] = dataset_class_dict
            dataset_info_dict[dataset_id] = dataset_img_type_dict
        info = pd.DataFrame(dataset_info_dict).stack().apply(pd.Series)
        info = info[sorted(info.columns)]
        info.fillna(0, inplace=True)
        save_path = os.path.join(MAIN_DIR, 'data_info', f'{self.args["mode"]}', f'{self.args["image_type"]}', f"{mode}_data_info")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        info.to_html(save_path + '.html', bold_rows=False, border=4)
        info.to_csv(save_path + '.csv')
        return dataset_info_dict

    def dataset_attributes(self):
        atr_dict = {"mode": self.args['mode'],
                    "classes": self.class_names, "num_classes": self.num_classes,
                    "train_class_samples": np.sum(self.train_data[1]['class'], axis=0), "train_len": len(self.train_data[1]['class']),
                    "val_class_samples": np.sum(self.val_data[1]['class'], axis=0), "val_len": len(self.val_data[1]['class']),
                    "test_class_samples": np.sum(self.test_data[1]['class'], axis=0), "test_len": len(self.test_data[1]['class']),
                    'weights_per_image_type': self.weights_per_image_type}
        return atr_dict

    def info(self):
        attr = self.dataset_attributes()
        output = f"Mode: {attr['mode']}\n" \
                 f"Classes: {attr['classes']}\n" \
                 f"Train Class Samples: {attr['train_class_samples']}\n" \
                 f"Train Length: {attr['train_len']}\n" \
                 f"Validation Class Samples: {attr['val_class_samples']}\n" \
                 f"Validation Length: {attr['val_len']}\n" \
                 f"Test Class Samples: {attr['test_class_samples']}\n" \
                 f"Test Length: {attr['test_len']}\n" \
                 f"Weights per class:{self.weights_per_class}\n"\
                 f"Sample weight by image type:{attr['weights_per_image_type']}\n"
        return output

    def tf_imread(self, sample, label, sample_weight, mode):
        sample["image"] = tf.reshape(tensor=tf.image.decode_image(tf.io.read_file(sample["image"]), channels=3), shape=self.args['input_shape'])
        sample["image"] = self.args['preprocess_fn'](sample["image"])
        if mode == "train":
            translation = tf.random.uniform(shape=[2], seed=self.random_state, minval=-20, maxval=20, dtype=tf.float32)
            random_degrees = tf.random.uniform(shape=[1], minval=0, seed=self.random_state, maxval=360, dtype=tf.float32)
            rand = tf.random.uniform(shape=[1], minval=0, seed=self.random_state, maxval=1., dtype=tf.float32)
            sigma = np.random.uniform(low=0., high=2.)  # tf.random.uniform(shape=[], minval=0, seed=random_state, maxval=2.5, dtype=tf.float32)
            sharp = tf.random.uniform(shape=[1], minval=0., seed=self.random_state, maxval=.2, dtype=tf.float32)
            sample["image"] = tf.image.random_flip_left_right(image=sample["image"], seed=self.random_state)
            sample["image"] = tf.image.random_flip_up_down(image=sample["image"], seed=self.random_state)
            sample["image"] = tf.image.random_brightness(image=sample["image"], max_delta=0.1, seed=self.random_state)
            sample["image"] = tf.image.random_contrast(image=sample["image"], lower=.5, upper=1.5, seed=self.random_state)
            sample["image"] = tf.image.random_saturation(image=sample["image"], lower=0.8, upper=1.2, seed=self.random_state)
            sample["image"] = tfa.image.translate(sample["image"], translations=translation, name="Translation")
            sample["image"] = tfa.image.rotate(images=sample["image"], angles=random_degrees, name="Rotation")
            sample["image"] = tfa.image.sharpness(image=tf.cast(sample["image"], dtype=tf.float32), factor=sharp, name="Sharpness")
            sample["image"] = tf.cond(tf.math.less_equal(rand, 0.5), lambda: tfa.image.gaussian_filter2d(image=sample["image"], sigma=sigma, filter_shape=5, name="Gaussian_filter"), lambda: sample["image"])
        return sample, label, sample_weight

    def get_dataset(self, mode=None, repeat=1):
        np.random.seed(self.random_state)

        if mode == "train":
            dataset = self.train_data
        elif mode == "val":
            dataset = self.val_data
        elif mode == "test":
            dataset = self.test_data
        else:
            raise ValueError(f"{mode} is not a valid mode.")
        dataset[0].pop("dataset_id")
        dataset[0].pop("image_type_weights")
        dataset[0].pop("class_weights")
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(lambda sample, label, sample_weight: self.tf_imread(sample, label, sample_weight, mode=mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    args = {'mode': '5cls', 'image_type': 'both', "dataset_frac": 0.1}
    dir_dict = directories(trial_id='1', run_num=0, args=args)
