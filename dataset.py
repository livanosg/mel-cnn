import os
from augmentations import TFAugmentations
from config import MAPPER, directories, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER, CLASS_NAMES
import tensorflow as tf
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, dir_dict: dict, input_shape: tuple, args: dict, batch: int = 8, ):
        self.random_state = 1312
        self.mode = args["mode"]
        self.frac = args["dataset_frac"]
        self.input_shape = input_shape
        self.batch = batch
        self.dir_dict = dir_dict
        self.image_folder = self.dir_dict["image_folder"]
        self.image_type = args["image_type"]
        self.class_names = CLASS_NAMES[self.mode]
        self.num_classes = len(self.class_names)
        self.features = pd.read_csv(self.dir_dict["data"])
        self.features.fillna(-10, inplace=True)
        self.features["age_approx"] -= (self.features["age_approx"] % 10)  # group at decades
        self.features = self.features.replace(to_replace=MAPPER)
        if self.mode == "ben_mal":
            self.features = self.features.replace(to_replace=BEN_MAL_MAPPER)
        elif self.mode == "nev_mel":
            self.features = self.features.replace(to_replace=NEV_MEL_OTHER_MAPPER)
        else:
            pass
        # --------------------------========== Remove duplicates from ISIC 2020 ===========--------------------------- #
        self.duplicates = pd.read_csv("ISIC_2020_Training_Duplicates.csv")
        self.duplicates["image_name_2"] = self.duplicates["image_name_2"] + ".jpg"
        self.features = self.features[~self.features["image"].isin(self.duplicates["image_name_2"])]
        # --------------------------=======================================================--------------------------- #
        if self.image_type != "both":
            self.features = self.features[self.features["image_type"] == MAPPER["image_type"][self.image_type]]
        else:
            # ------------------------============== Calculate Sample weights ===============------------------------- #
            value_counts = self.features["image_type"].value_counts(sort=False, ascending=True)
            self.weight_by_type = np.sum(value_counts) / np.asarray([value_counts[0], value_counts[1]])
            self.features["sample_weights"] = np.where(self.features["image_type"] == "clinic", self.weight_by_type[0],
                                                       self.weight_by_type[1])
            # ------------------------=======================================================------------------------- #
        self.features["image"] = f"{self.image_folder}{os.sep}data{os.sep}" + self.features["dataset_id"] + f"{os.sep}data{os.sep}" + self.features["image"]
        self.train_data, self.val_data, self.test_data = map(self.ohe_map, self.split_data())
        self.attr = self.dataset_attributes()

    def split_data(self):
        """Split data to train and val with a train_frac ratio. Also returns a fraction of initial dataset.
         Inputs: train_frac: fraction of (fraction of) data to be used for training.
         frac: fraction of initial data to be used.
         Outputs: a pair of (train data, validation data) dicts."""
        train_data = []
        val_data = []
        # Keep "up" + "7pt" for testing
        if self.mode in ["ben_mal", "nev_mel"]:  # drop Suspicious_benign from ben_mal and nev_mel
            self.features = self.features[self.features["class"] != 2]
        if self.mode in ["5cls", "nev_mel"]:  # drop unknown_benign from 5cls and nev_mel
            self.features = self.features[self.features["class"] != 5]
        test_data = self.features[self.features["dataset_id"].isin(["up", "7pt"])]
        rest_data = self.features[~self.features.index.isin(test_data.index)]

        for _image_type in rest_data["image_type"].unique():
            image_type_data = rest_data[rest_data.loc[:, "image_type"] == _image_type]
            for _class in range(self.num_classes):
                class_data = image_type_data[image_type_data.loc[:, "class"] == _class]  # fraction per class
                train_data_class_frac = class_data.sample(frac=0.8, random_state=self.random_state)  # 80% for training
                rest_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
                train_data.append(train_data_class_frac)
                val_data.append(rest_data_class_frac)
        train_data = pd.concat(train_data, axis=0).sample(frac=self.frac, random_state=self.random_state)
        val_data = pd.concat(val_data, axis=0).sample(frac=self.frac, random_state=self.random_state)
        # test_data = pd.concat(test_data)
        return dict(train_data), dict(val_data), dict(test_data)

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
        if self.image_type == "both":
            sample_weights = features.pop("sample_weights")
            return features, labels, sample_weights
        else:
            return features, labels

    def per_dataset_info(self):
        dataset_info_dict = {}
        image_type_inv = {}
        for (key, value) in MAPPER["image_type"].items():
            image_type_inv[value] = key
        for dataset_id in self.features["dataset_id"].unique():
            dataset_part = self.features[self.features.loc[:, "dataset_id"] == dataset_id]  # fraction per class
            print(f"{dataset_id}: {len(dataset_part)}")
            dataset_img_type_dict = {}
            _image_type = dataset_part["image_type"].unique()
            for i in _image_type:
                dataset_image_part = dataset_part[dataset_part.loc[:, "image_type"] == i]
                class_counts = dataset_image_part["class"].value_counts()
                print(class_counts)
                dataset_class_dict = {}
                for j in class_counts.keys():
                    dataset_class_dict[CLASS_NAMES[self.mode][j]] = class_counts[j]
                dataset_img_type_dict[image_type_inv[i]] = dataset_class_dict
            dataset_info_dict[dataset_id] = dataset_img_type_dict
        info = pd.DataFrame(dataset_info_dict).stack().apply(pd.Series)
        info.sort_index(axis=0, level=0, inplace=True)
        info.fillna(0, inplace=True)
        info.to_html(os.path.join(self.dir_dict["main"], f"{self.mode}-data_info.html"))
        info.to_csv(os.path.join(self.dir_dict["main"], f"{self.mode}-data_info.csv"))
        return dataset_info_dict

    def info(self):
        attr = self.dataset_attributes()
        return f"Mode: {attr['mode']}\n" \
               f"Classes: {attr['classes']}\n" \
               f"Train Class Samples: {attr['train_class_samples']}\n" \
               f"Train Length: {attr['train_len']}\n" \
               f"Validation Class Samples: {attr['val_class_samples']}\n" \
               f"Validation Length: {attr['val_len']}\n" \
               f"Test Class Samples: {attr['test_class_samples']}\n" \
               f"Test Length: {attr['test_len']}\n" \
               f"Weights per class:{self.get_class_weights()}\n"

    def dataset_attributes(self):
        return {"mode": self.mode,
                "classes": self.class_names,
                "num_classes": self.num_classes,
                "train_class_samples": np.sum(self.train_data[1]['class'], axis=0),
                "train_len": len(self.train_data[1]['class']),
                "val_class_samples": np.sum(self.val_data[1]['class'], axis=0),
                "val_len": len(self.val_data[1]['class']),
                "test_class_samples": np.sum(self.test_data[1]['class'], axis=0),
                "test_len": len(self.test_data[1]['class']),
                }

    def get_class_weights(self):
        return np.divide(self.attr["train_len"],
                         np.multiply(np.sum(self.train_data[1]["class"], axis=0), self.attr["num_classes"]))

    def get_dataset(self, mode=None, repeat=1):
        random_state = self.random_state
        input_shape = self.input_shape
        if mode == "train":
            dataset = self.train_data
        elif mode == "val":
            dataset = self.val_data
        elif mode == "test":
            dataset = self.test_data
        else:
            raise ValueError(f"{mode} is not a valid mode.")

        def tf_imread(*ds):
            ds[0]["image"] = tf.reshape(tensor=tf.image.decode_image(tf.io.read_file(ds[0]["image"]), channels=3), shape=input_shape)
            if mode == "train":
                tf_augms = TFAugmentations(random_state)
                ds[0]["image"] = tf_augms.augm(ds[0]["image"])
            return ds

        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(tf_imread, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    test_args = {"mode": "5cls",
                 "dataset_frac": 1,
                 "image_type": "both",
                 "image_size": 100}
    inpt_shape = (test_args["image_size"], test_args["image_size"], 3)
    test_dir_dict = directories(trial_id=1, run_num=0, img_size=100, colour="rgb", args=test_args)
    a = MelData(batch=1, dir_dict=test_dir_dict, args=test_args, input_shape=inpt_shape)
