import cv2.cv2 as cv2
from augmentations import Augmentations
from config import MAPPER, directories, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER
import tensorflow as tf
import numpy as np
import pandas as pd


# todo pick up + ? dataset for test
class MelData:
    def __init__(self, file: str, trial: str, frac: float = 1., img_folder: str = None, batch: int = None,
                 mode="5cls", ):
        assert mode in ("5cls", "ben_mal", "nev_mel")
        self.random_state = 1312
        self.mode = mode
        self.trial = trial
        self.batch = batch
        self.frac = frac
        if self.mode == "5cls":
            self.classes = 5
        else:
            self.classes = 2
        self.features = pd.read_csv(file)
        self.features.fillna(-10, inplace=True)
        self.features["image"] = "data/" + self.features["dataset_id"] + "/data/" + self.features["image"]
        self.features.drop("dataset_id", axis=1, inplace=True)
        self.features["age_approx"] = self.features["age_approx"] - (
                self.features["age_approx"] % 10)  # group at decades
        self.features.replace(to_replace=MAPPER, inplace=True)
        self.features.rename(columns={"diagnosis": "class"}, inplace=True)
        self.features["image"] = img_folder + "/" + self.features["image"]
        values = self.features["image_type"].value_counts()
        self.weight_by_type = np.sum(values) / np.asarray(values)
        self.train_data, self.val_data, self.test_data = map(self.ohe_map, self.split_data(frac=self.frac))

    def split_data(self, frac):
        """Split data to train and val with a train_frac ratio. Also returns a fraction of initial dataset.
         Inputs: train_frac: fraction of (fraction of) data to be used for training.
         frac: fraction of initial data to be used.
         Outputs: a pair of (train data, validation data) dicts."""
        train_data = []
        val_data = []
        test_data = []
        if self.mode == "ben_mal":
            MAPPER["class"] = BEN_MAL_MAPPER["class"]
        elif self.mode == "nev_mel":
            MAPPER["class"] = NEV_MEL_OTHER_MAPPER["class"]
        if self.mode in ["ben_mal", "nev_mel"]:  # drop Suspicious_benign from ben_mal and nev_mel
            self.features.replace(to_replace=MAPPER, inplace=True)
            self.features = self.features[self.features["class"] != 2]
        if self.mode in ["5cls", "nev_mel"]:  # drop unknown_benign from 5cls and nev_mel
            self.features = self.features[self.features["class"] != 5]

        for _class in range(self.classes):
            class_data = self.features[self.features.loc[:, "class"] == _class]  # fraction per class
            train_data_class_frac = class_data.sample(frac=0.7, random_state=self.random_state)  # 70% for training
            test_val_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
            val_data_frac = test_val_data_class_frac.sample(frac=0.5, random_state=self.random_state)  # 15% validation
            test_data_frac = test_val_data_class_frac[
                ~test_val_data_class_frac.index.isin(val_data_frac.index)]  # 15% test
            train_data.append(train_data_class_frac.sample(frac=frac, random_state=self.random_state))
            val_data.append(val_data_frac.sample(frac=frac, random_state=self.random_state))
            test_data.append(test_data_frac.sample(frac=frac, random_state=self.random_state))
        train_data = pd.concat(train_data).sample(frac=1., random_state=self.random_state)
        val_data = pd.concat(val_data).sample(frac=1., random_state=self.random_state)
        test_data = pd.concat(test_data).sample(frac=1., random_state=self.random_state)
        return dict(train_data), dict(val_data), dict(test_data)

    def ohe_map(self, features):
        """ Turn features to one-hot encoded vectors.
        Inputs:
            features: dictionary of features int encoded.
        Outputs:
            (features, labels) dicts.
        """

        for key in features.keys():
            if key == "class":
                features[key] = tf.keras.backend.one_hot(indices=np.asarray(features[key]), num_classes=self.classes)
            elif key not in ["image", "dataset_id"]:
                unique_vales = self.features[key].unique()
                if -1 in unique_vales:
                    num_classes = len(unique_vales) - 1
                else:
                    num_classes = len(unique_vales)
                features[key] = tf.keras.backend.one_hot(indices=np.asarray(features[key]), num_classes=num_classes)
            else:
                pass
        labels = {"class": features.pop("class")}
        return features, labels

    @staticmethod
    def data_augm(filename):
        filename = cv2.imread(filename.numpy().decode('ascii'))
        augms = Augmentations()
        filename = augms(input_image=filename)
        return filename

    def train_imread(self, features, labels):
        """Read image from (features, labels) dicts and return"""
        features["image"] = tf.py_function(self.data_augm, [features["image"]], tf.dtypes.uint8)
        # features["image"] = tf.image.decode_image(tf.io.read_file(features["image"]), channels=3)
        return features, labels

    @staticmethod
    def val_imread(features, labels):
        """Read image from (features, labels) dicts and return"""
        features["image"] = tf.py_function(lambda x: cv2.imread(x.numpy().decode('ascii')), [features["image"]],
                                           tf.dtypes.uint8)
        # features["image"] = tf.image.decode_image(tf.io.read_file(features["image"]), channels=3)
        return features, labels

    def data_info(self):
        return {"train_len": len(self.train_data[1]["class"]),
                "val_len": len(self.val_data[1]["class"]),
                "test_len": len(self.test_data[1]["class"])}

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
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_class_weights(self):
        return np.divide(self.data_info()["train_len"],
                         np.multiply(np.sum(self.train_data[1]["class"], axis=0), self.classes))


if __name__ == '__main__':
    img_fldr = directories(trial_id=1, mode="nev_mel", run_num=0, img_size=224, colour="rgb")["image_folder"]
    a = MelData("all_data.csv", frac=0.1, img_folder=img_fldr, batch=1, mode="nev_mel", trial="test/")
    b = a.get_dataset(mode="train", repeat=1)
    print(a.get_class_weights())
    print(a.weight_by_type)
    j = 0
    for i in b.as_numpy_iterator():
        print(i[1]["class"])  # image_type sex age_approx anatom_site_general
        j += 1
        if j > 10:
            break
