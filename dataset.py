import os
import cv2
from augmentations import Augmentations
from config import MAPPER, directories, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER, CLASS_NAMES
import tensorflow as tf
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, file: str, dir_dict: dict, args: dict, batch: int = None):
        self.random_state = 1312
        self.mode = args["mode"]
        self.frac = args["dataset_frac"]
        print(self.frac)
        self.batch = batch
        self.image_folder = dir_dict["image_folder"]
        self.image_type = args["image_type"]
        self.classes = CLASS_NAMES[self.mode]
        self.num_classes = len(self.classes)
        self.features = pd.read_csv(file)
        # --------------------------========== Remove duplicates from ISIC 2020 ===========--------------------------- #
        self.duplicates = pd.read_csv("ISIC_2020_Training_Duplicates.csv")
        self.duplicates["image_name_2"] = self.duplicates["image_name_2"] + ".jpg"
        self.features = self.features[~self.features["image"].isin(self.duplicates["image_name_2"])]
        # --------------------------=======================================================--------------------------- #
        if self.image_type != "both":
            self.features = self.features[self.features["image_type"] == self.image_type]
        else:
            # ------------------------============== Calculate Sample weights ===============------------------------- #
            value_counts = self.features["image_type"].value_counts(sort=False, ascending=True)
            print(value_counts["clinic"])
            self.weight_by_type = np.sum(value_counts) / np.asarray([value_counts["clinic"], value_counts["derm"]])
            self.features["sample_weights"] = np.where(self.features["image_type"] == "clinic", self.weight_by_type[0], self.weight_by_type[1])
            # ------------------------=======================================================------------------------- #
        self.features.fillna(-10, inplace=True)
        self.features["image"] = "data" + os.sep + self.features["dataset_id"] + os.sep + "data" + os.sep + self.features["image"]
        self.features["age_approx"] -= (self.features["age_approx"] % 10)  # group at decades
        self.features.replace(to_replace=MAPPER, inplace=True)
        self.features["image"] = self.image_folder + os.sep + self.features["image"]
        self.train_data, self.val_data, self.test_data = map(self.ohe_map, self.split_data(frac=self.frac))

    def split_data(self, frac):
        """Split data to train and val with a train_frac ratio. Also returns a fraction of initial dataset.
         Inputs: train_frac: fraction of (fraction of) data to be used for training.
         frac: fraction of initial data to be used.
         Outputs: a pair of (train data, validation data) dicts."""
        train_data = []
        val_data = []
        test_data = []
        # if self.image_type == "clinic":
        #     test_data = self.features[self.features["dataset_id"].isin(["7pt"])]  # Keep 7pt and up for test
        # else:
        #     test_data = self.features[self.features["dataset_id"].isin(["7pt", "up"])]  # Keep 7pt and up for test
        # self.features.drop(test_data.index, inplace=True)

        if self.mode == "ben_mal":
            MAPPER["class"] = BEN_MAL_MAPPER["class"]
        elif self.mode == "nev_mel":
            MAPPER["class"] = NEV_MEL_OTHER_MAPPER["class"]
        if self.mode in ["ben_mal", "nev_mel"]:  # drop Suspicious_benign from ben_mal and nev_mel
            self.features.replace(to_replace=MAPPER, inplace=True)
            self.features = self.features[self.features["class"] != 2]
        if self.mode in ["5cls", "nev_mel"]:  # drop unknown_benign from 5cls and nev_mel
            self.features = self.features[self.features["class"] != 5]

        for _image_type in range(2):
            image_type_data = self.features[self.features.loc[:, "image_type"] == _image_type]  # fraction per class
            for _class in range(self.num_classes):
                class_data = image_type_data[image_type_data.loc[:, "class"] == _class]  # fraction per class
                train_data_class_frac = class_data.sample(frac=0.8, random_state=self.random_state)  # 80% for training
                train_data.append(train_data_class_frac)
                rest_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
                val_data_class_frac = rest_data_class_frac[:len(rest_data_class_frac)//2]
                test_data_class_frac = rest_data_class_frac[len(rest_data_class_frac)//2:]
                val_data.append(val_data_class_frac)
                test_data.append(test_data_class_frac)
        train_data = pd.concat(train_data).sample(frac=frac, random_state=self.random_state).drop("dataset_id", axis=1)
        val_data = pd.concat(val_data).sample(frac=frac, random_state=self.random_state).drop("dataset_id", axis=1)
        test_data = pd.concat(test_data).drop("dataset_id", axis=1)
        return dict(train_data), dict(val_data), dict(test_data)

    def ohe_map(self, features):
        """ Turn features to one-hot encoded vectors.
        Inputs:
            features: dictionary of features int encoded.
        Outputs:
            (features, labels) dicts.
        """

        features["image_type"] = tf.keras.backend.one_hot(indices=np.asarray(features["image_type"]), num_classes=2)
        features["sex"] = tf.keras.backend.one_hot(indices=np.asarray(features["sex"]), num_classes=2)
        features["age_approx"] = tf.keras.backend.one_hot(indices=np.asarray(features["age_approx"]), num_classes=10)
        features["anatom_site_general"] = tf.keras.backend.one_hot(indices=np.asarray(features["anatom_site_general"]), num_classes=6)
        features["class"] = tf.keras.backend.one_hot(indices=np.asarray(features["class"]), num_classes=self.num_classes)
        labels = {"class": features.pop("class")}
        if self.image_type == "both":
            sample_weights = features.pop("sample_weights")
            return features, labels, sample_weights
        else:
            return features, labels

    @staticmethod
    def data_augm(filename):
        filename = cv2.imread(filename.numpy().decode('ascii'))
        augms = Augmentations()
        filename = augms(input_image=filename)
        return filename

    def train_imread(self, *args):
        """Read image from (features, labels) dicts and return"""
        args[0]["image"] = tf.py_function(self.data_augm, [args[0]["image"]], tf.dtypes.uint8)
        return args

    @staticmethod
    def val_imread(*args):
        """Read image from (features, labels) dicts and return"""
        args[0]["image"] = tf.py_function(lambda x: cv2.imread(x.numpy().decode('ascii')), [args[0]["image"]], tf.dtypes.uint8)
        return args

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

    def datasets_info(self):
        for dataset in self.features["dataset_id"].unique():
            dataset_split = self.features[self.features.loc[:, "dataset_id"] == dataset]
            prnt_str = f"{dataset}\n"
            for idx, value in dict(dataset_split["class"].value_counts(sort=False, ascending=True)).items():
                prnt_str += f"{self.classes[idx]} : {value}\n"
            prnt_str += f"{10 * '-'}\n"

            image_type = {0: "clinic",
                          1: "derm"}
            for idx, value in dict(dataset_split["image_type"].value_counts(sort=False, ascending=True)).items():
                prnt_str += f"{image_type[idx]} : {value}\n"

            print(prnt_str)

    def dataset_attributes(self):
        return {"mode": self.mode,
                "classes": self.classes,
                "train_class_samples": np.sum(self.train_data[1]['class'], axis=0),
                "train_len": len(self.train_data[1]['class']),
                "val_class_samples": np.sum(self.val_data[1]['class'], axis=0),
                "val_len": len(self.val_data[1]['class']),
                "test_class_samples": np.sum(self.test_data[1]['class'], axis=0),
                "test_len": len(self.test_data[1]['class']),
                }

    def get_class_weights(self):
        """Class-Balanced Loss Based on Effective Number of Samples
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        """
        beta = 0.9999
        attr = self.dataset_attributes()
        effective_num = 1.0 - np.power(beta, attr["train_class_samples"])
        weights_for_samples = (1.0 - beta) / np.array(effective_num)
        weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * len(attr["classes"])
        return weights_for_samples

    def get_dataset(self, mode=None, repeat=1):
        if mode == "train":
            dataset = self.train_data
        elif mode == "val":
            dataset = self.val_data
        elif mode == "test":
            dataset = self.test_data
        else:
            raise ValueError(f"{mode} is not a valid mode.")
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        if mode == "train":
            dataset = dataset.map(self.train_imread, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if mode in ["val", "test"]:
            dataset = dataset.map(self.val_imread, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    args = {"mode": "5cls",
            "dataset_frac": 1,
            "image_type": "both"}

    dir_dict = directories(trial_id=1, run_num=0, img_size=100, colour="rgb", args=args)
    a = MelData(file="all_data.csv", batch=50, dir_dict=dir_dict, args=args)
    b = a.get_dataset(mode="train", repeat=1)
    print(a.info())
    # a.datasets_info()
    # for data in b.as_numpy_iterator():
    #     print(data)
    #     break
