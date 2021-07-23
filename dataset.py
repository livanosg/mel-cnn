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
        self.image_folder = dir_dict["image_folder"]
        self.image_type = args["image_type"]
        self.class_names = CLASS_NAMES[self.mode]
        self.num_classes = len(self.class_names)
        self.features = pd.read_csv(dir_dict["data"])
        self.features.fillna(-10, inplace=True)
        self.features["age_approx"] -= (self.features["age_approx"] % 10)  # group at decades
        self.features.replace(to_replace=MAPPER, inplace=True)
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
        # Keep "up" for testing
        test_data = [self.features[self.features["dataset_id"] == "up"]]
        self.features = self.features[~self.features.index.isin(self.features[self.features["dataset_id"] == "up"].index)]
        if self.mode == "ben_mal":
            MODE_MAPPER = BEN_MAL_MAPPER["class"]
        elif self.mode == "nev_mel":
            MODE_MAPPER = NEV_MEL_OTHER_MAPPER["class"]
        else:
            MODE_MAPPER = None
        if self.mode in ["ben_mal", "nev_mel"]:  # drop Suspicious_benign from ben_mal and nev_mel
            self.features = self.features.replace(to_replace=MODE_MAPPER)
            self.features = self.features[self.features["class"] != 2]
        if self.mode in ["5cls", "nev_mel"]:  # drop unknown_benign from 5cls and nev_mel
            self.features = self.features[self.features["class"] != 5]

        for _image_type in self.features["image_type"].unique():
            image_type_data = self.features[self.features.loc[:, "image_type"] == _image_type]
            for _class in range(self.num_classes):
                class_data = image_type_data[image_type_data.loc[:, "class"] == _class]  # fraction per class
                train_data_class_frac = class_data.sample(frac=0.8, random_state=self.random_state)  # 80% for training
                rest_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
                val_data_class_frac = rest_data_class_frac.sample(frac=0.5, random_state=self.random_state)
                test_data_class_frac = rest_data_class_frac[~rest_data_class_frac.index.isin(val_data_class_frac.index)]
                train_data.append(train_data_class_frac)
                val_data.append(val_data_class_frac)
                test_data.append(test_data_class_frac)
        train_data = pd.concat(train_data).sample(frac=self.frac, random_state=self.random_state).drop("dataset_id", axis=1)
        val_data = pd.concat(val_data).sample(frac=self.frac, random_state=self.random_state).drop("dataset_id", axis=1)
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
        features["anatom_site_general"] = tf.keras.backend.one_hot(indices=np.asarray(features["anatom_site_general"]),
                                                                   num_classes=6)
        features["class"] = tf.keras.backend.one_hot(indices=np.asarray(features["class"]),
                                                     num_classes=self.num_classes)
        labels = {"class": features.pop("class")}
        if self.image_type == "both":
            sample_weights = features.pop("sample_weights")
            return features, labels, sample_weights
        else:
            return features, labels

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
        """Class-Balanced Loss Based on Effective Number of Samples
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        """
        # beta = 0.9999
        # effective_num = 1.0 - np.power(beta, attr["train_class_samples"])
        # weights_for_samples = (1.0 - beta) / np.array(effective_num)
        # weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * len(attr["classes"])
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
    import cv2
    test_args = {"mode": "5cls",
                 "dataset_frac": 1,
                 "image_type": "both",
                 "image_size": 100}
    inpt_shape = (test_args["image_size"], test_args["image_size"], 3)
    test_dir_dict = directories(trial_id=1, run_num=0, img_size=100, colour="rgb", args=test_args)
    a = MelData(batch=1, dir_dict=test_dir_dict, args=test_args, input_shape=inpt_shape)
    b = a.get_dataset(mode="val", repeat=1)
    cv2.namedWindow("sample", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("sample_norm", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("path_data", cv2.WINDOW_FREERATIO)

    def yield_data():
        df = pd.DataFrame.from_dict(a.val_data[0])
        for i in df.iterrows():
            yield i

    for data, paths in zip(b.as_numpy_iterator(), yield_data()):
        img = (data[0]["image"][0, ...] - np.min(data[0]["image"][0, ...])) / (np.max(data[0]["image"][0, ...]) - np.min(data[0]["image"][0, ...]))
        img = cv2.cvtColor(np.array(img * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imshow("sample_norm", img)
        cv2.imshow("sample", np.array(data[0]["image"][0, ...]).astype(np.uint8))
        cv2.imshow("path_data", cv2.imread(paths[1]["image"]))
        cv2.waitKey()
