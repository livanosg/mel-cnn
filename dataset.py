import os

import cv2

from config import MAPPER, directories, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER, CLASS_NAMES, MAIN_DIR
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, dir_dict: dict, args: dict, batch: int = 8):
        self.random_state = 1312
        self.args = args
        self.batch = batch
        self.dir_dict = dir_dict
        self.image_folder = self.dir_dict["image_folder"]
        self.class_names = CLASS_NAMES[self.args['mode']]
        self.num_classes = len(self.class_names)
        self.train_data_df = pd.read_csv(self.dir_dict["data_csv"]["train"])
        self.val_data_df = pd.read_csv(self.dir_dict["data_csv"]["val"])
        self.test_data_df = pd.read_csv(self.dir_dict["data_csv"]["test"])
        # ------------------------================ Calculate Sample weights =================------------------------- #
        if self.args["image_type"] == "both":
            value_counts = dict(self.train_data_df["image_type"].value_counts(sort=False, ascending=True))
            self.weight_by_type = np.sum(list(value_counts.values())) / np.multiply(len(value_counts),
                                                                                    [value_counts[0], value_counts[1]])
        # ------------------------===========================================================------------------------- #
        self.train_data = self.ohe_map(dict(self.prep_classes(self.train_data_df).sample(frac=self.args["dataset_frac"])))
        self.val_data = self.ohe_map(dict(self.prep_classes(self.val_data_df).sample(frac=self.args["dataset_frac"])))
        self.test_data = self.ohe_map(dict(self.prep_classes(self.test_data_df)))
        self.attr = self.dataset_attributes()

    def prep_classes(self, data):
        data["image"] = f"{self.image_folder}{os.sep}" + data["image"]
        if self.args["image_type"] == "both":
            data["sample_weights"] = np.where(data["image_type"] == "clinic", self.weight_by_type[0], self.weight_by_type[1])
        if self.args['mode'] == "ben_mal":
            data.replace(to_replace=BEN_MAL_MAPPER, inplace=True)
        if self.args['mode'] == "nev_mel":
            data.replace(to_replace=NEV_MEL_OTHER_MAPPER, inplace=True)
        if self.args["image_type"] != "both":  # Keep derm or clinic, samples.
            data = data[data["image_type"] == MAPPER["image_type"][self.args["image_type"]]]
        if self.args['mode'] in ["ben_mal", "nev_mel"]:
            data = data[data["class"] != 2]
        if self.args['mode'] in ["5cls"]:
            data = data[data["class"] != 5]
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
        features.pop("dataset_id")
        if self.args["image_type"] == "both":
            sample_weights = features.pop("sample_weights")
            return features, labels, sample_weights
        else:
            return features, labels

    def data_info(self, mode):
        dataset_info_dict = {}
        image_type_inv = {}
        if mode == "all":
            df = self.train_data_df.append(self.val_data_df).append(self.test_data_df)
        elif mode == "train":
            df = self.train_data_df

        elif mode == "val":
            df = self.val_data_df
        else:
            df = self.test_data_df
        for (key, value) in MAPPER["image_type"].items():
            image_type_inv[value] = key
        for dataset_id in df["dataset_id"].unique():
            dataset_part = df[df.loc[:, "dataset_id"] == dataset_id]  # fraction per class
            dataset_img_type_dict = {}
            image_types = dataset_part["image_type"].unique()
            for image_type in image_types:
                dataset_image_part = dataset_part[dataset_part.loc[:, "image_type"] == image_type]
                class_counts = dataset_image_part["class"].value_counts()
                dataset_class_dict = {}
                for j in class_counts.keys():
                    dataset_class_dict[CLASS_NAMES[self.args['mode']][j]] = class_counts[j]
                dataset_img_type_dict[image_type_inv[image_type]] = dataset_class_dict
            dataset_info_dict[dataset_id] = dataset_img_type_dict
        info = pd.DataFrame(dataset_info_dict).stack().apply(pd.Series)
        info.sort_index(axis=0, level=0, inplace=True)
        info = info[sorted(info.columns)]
        info.fillna(0, inplace=True)
        save_path = os.path.join(MAIN_DIR, "data_info", f"{mode}_{self.args['image_type']}-{self.args['mode']}_data_info.html")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        info.to_html(save_path, bold_rows=False, border=5)
        return dataset_info_dict

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
                 f"Weights per class:{self.weights_per_class()}\n"
        if self.args["image_type"] == "both":
            output += f"Sample weight by image type:{attr['sample_weights']}\n"
        return output

    def dataset_attributes(self):
        atr_dict = {"mode": self.args['mode'],
                    "classes": self.class_names, "num_classes": self.num_classes,
                    "train_class_samples": np.sum(self.train_data[1]['class'], axis=0), "train_len": len(self.train_data[1]['class']),
                    "val_class_samples": np.sum(self.val_data[1]['class'], axis=0), "val_len": len(self.val_data[1]['class']),
                    "test_class_samples": np.sum(self.test_data[1]['class'], axis=0), "test_len": len(self.test_data[1]['class'])}
        if self.args["image_type"] == "both":
            atr_dict["sample_weights"] = self.weight_by_type
        return atr_dict

    def weights_per_class(self):
        return np.divide(self.attr["train_len"],
                         np.multiply(self.attr['train_class_samples'], self.attr['num_classes']))

    def tf_imread(self, *ds, mode):
        ds[0]["image"] = tf.reshape(tensor=tf.image.decode_image(tf.io.read_file(ds[0]["image"]), channels=3), shape=self.args['input_shape'])
        ds[0]["image"] = self.args["preprocess_fn"](ds[0]["image"])
        if mode == "train":
            translation = tf.random.uniform(shape=[2], seed=self.random_state, minval=-20, maxval=20, dtype=tf.float32)
            random_degrees = tf.random.uniform(shape=[1], minval=0, seed=self.random_state, maxval=360, dtype=tf.float32)
            rand = tf.random.uniform(shape=[1], minval=0, seed=self.random_state, maxval=1., dtype=tf.float32)
            sigma = np.random.uniform(low=0., high=2.)  # tf.random.uniform(shape=[], minval=0, seed=random_state, maxval=2.5, dtype=tf.float32)
            sharp = tf.random.uniform(shape=[1], minval=0., seed=self.random_state, maxval=.2, dtype=tf.float32)
            ds[0]["image"] = tf.image.random_flip_left_right(image=ds[0]["image"], seed=self.random_state)
            ds[0]["image"] = tf.image.random_flip_up_down(image=ds[0]["image"], seed=self.random_state)
            ds[0]["image"] = tf.image.random_brightness(image=ds[0]["image"], max_delta=0.1, seed=self.random_state)
            ds[0]["image"] = tf.image.random_contrast(image=ds[0]["image"], lower=.5, upper=1.5, seed=self.random_state)
            ds[0]["image"] = tf.image.random_saturation(image=ds[0]["image"], lower=0.8, upper=1.2, seed=self.random_state)
            ds[0]["image"] = tfa.image.translate(ds[0]["image"], translations=translation, name="Translation")
            ds[0]["image"] = tfa.image.rotate(images=ds[0]["image"], angles=random_degrees, name="Rotation")
            ds[0]["image"] = tfa.image.sharpness(image=tf.cast(ds[0]["image"], dtype=tf.float32), factor=sharp, name="Sharpness")
            ds[0]["image"] = tf.cond(tf.math.less_equal(rand, 0.5), lambda: tfa.image.gaussian_filter2d(image=ds[0]["image"], sigma=sigma, filter_shape=5, name="Gaussian_filter"), lambda: ds[0]["image"])
        return ds

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

        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        if self.args["image_type"] == "both":
            dataset = dataset.map(lambda train, label, samples: self.tf_imread(train, label, samples, mode=mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(lambda train, label: self.tf_imread(train, label, mode=mode), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    test_args = {"mode": "nev_mel",
                 "dataset_frac": 1,
                 "image_type": "both",
                 "image_size": 224,
                 "colour": "rgb"}
    test_args['input_shape'] = (test_args["image_size"], test_args["image_size"], 3)
    test_dir_dict = directories(trial_id=1, run_num=0, args=test_args)
    a = MelData(batch=1, dir_dict=test_dir_dict, args=test_args)
    # a.data_info("all")
    # a.data_info("train")
    # a.data_info("val")
    # a.data_info("test")
    cv2.namedWindow("trial", cv2.WINDOW_FREERATIO)
    for i in a.get_dataset("train").as_numpy_iterator():
        print(i)
        # img = i[0]["image"] # [0, ...]
        # cv2.imshow("trial", cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        break
