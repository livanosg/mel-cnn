import cv2

from augmentations import Augmentations
from config import MAPPER, directories, BEN_MAL_MAPPER, NEV_MEL_OTHER_MAPPER
import tensorflow as tf
import numpy as np
import pandas as pd

# TODO select benign, malignant, 5classes or nevus, melanoma outputs
class MelData:
    def __init__(self, file: str, frac: float = 1., img_folder: str = None, batch: int = None, classes=5):
        assert classes in (2, 5)
        self.batch = batch
        self.frac = frac
        self.classes = classes
        self.features = pd.read_csv(file, dtype="str")
        self.features.pop("dataset_id")
        self.features.fillna(-10, inplace=True)
        self.features.replace(to_replace=MAPPER, inplace=True)
        self.features["image"] = img_folder + "/" + self.features["image"]
        self.train_data, self.val_data = map(self.ohe_map, self.split_data(train_frac=0.8, frac=self.frac))

    def split_data(self, train_frac, frac):
        """Split data to train and val with a train_frac ratio. Also returns a fraction of initial dataset.
         Inputs: train_frac: fraction of (fraction of) data to be used for training.
         frac: fraction of initial data to be used.
         Outputs: a pair of (train data, validation data) dicts."""
        train_data = []
        val_data = []
        for key in MAPPER["class"].keys():
            class_data = self.features[self.features.loc[:, "class"] == MAPPER["class"][key]]
            train_data_class_frac = class_data.sample(frac=train_frac, random_state=1312)
            val_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
            train_data.append(train_data_class_frac.sample(frac=frac, random_state=1312))
            val_data.append(val_data_class_frac.sample(frac=frac, random_state=1312))
        train_data = pd.concat(train_data).sample(frac=1., random_state=1312)
        val_data = pd.concat(val_data).sample(frac=1., random_state=1312)
        if self.classes == 2:
            MAPPER["class"] = BEN_MAL_MAPPER["class"]
            train_data.replace(to_replace=MAPPER, inplace=True)
            val_data.replace(to_replace=MAPPER, inplace=True)
        if self.classes == 3:
            MAPPER["class"] = NEV_MEL_OTHER_MAPPER["class"]
            train_data.replace(to_replace=MAPPER, inplace=True)
            val_data.replace(to_replace=MAPPER, inplace=True)

        return dict(train_data), dict(val_data)

    def ohe_map(self, features):
        """ Turn features to one-hot encoded vectors. Also return a pair of (features, labels) dicts.
        Inputs:
        features: dictionary of features int encoded."""

        for key in features.keys():
            if key == "class":
                features[key] = tf.keras.backend.one_hot(indices=np.asarray(features[key]), num_classes=self.classes)
            elif key != "image":
                features[key] = tf.keras.backend.one_hot(indices=np.asarray(features[key]), num_classes=len(MAPPER[key]))
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

    def imread(self, features, labels):
        """Read image from (features, labels) dicts and return"""
        features["image"] = tf.py_function(self.data_augm, [features["image"]], tf.dtypes.uint8)
        # features["image"] = tf.image.decode_image(tf.io.read_file(features["image"]), channels=3)
        return features, labels

    def data_info(self):
        return {"train_len": len(self.train_data[1]["class"]),
                "val_len": len(self.val_data[1]["class"])}

    def get_dataset(self, mode=None, repeat=None):
        if mode == "train":
            dataset = self.train_data
        elif mode == "val":
            dataset = self.val_data
        else:
            raise ValueError(f"{mode} is not a valid mode.")
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(self.imread, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    img_fldr = directories(run_num=0, img_size=224, colour="rgb")["image_folder"]
    a = MelData("all_data_init.csv", frac=0.1, img_folder=img_fldr, batch=1, classes=2)
    b = a.get_dataset(mode="train", repeat=1)
    print(a.get_class_weights())
    for i in b.as_numpy_iterator():
        print(i)
        break
