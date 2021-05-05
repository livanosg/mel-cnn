from config import MAPPER
import tensorflow as tf
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, file, frac=1., colour=None, hw=None, batch=None):
        self.batch = batch
        self.frac = frac
        self.features = pd.read_csv(file, dtype="str")
        self.features.pop("dataset_id")
        self.features.fillna(-10, inplace=True)
        self.features.replace(to_replace=MAPPER, inplace=True)
        self.features["image"] = f"proc_{str(hw)}_{colour}/" + self.features["image"]
        self.train_data, self.val_data = self.split_data(frac=self.frac)

    def split_data(self, frac):
        train_data = []
        val_data = []
        for key in MAPPER["class"].keys():
            class_data = self.features[self.features.loc[:, "class"] == MAPPER["class"][key]]
            train_data_class_frac = class_data.sample(frac=0.8, random_state=1312)
            val_data_class_frac = class_data[~class_data.index.isin(train_data_class_frac.index)]
            train_data.append(train_data_class_frac.sample(frac=frac, random_state=1312))
            val_data.append(val_data_class_frac.sample(frac=frac, random_state=1312))
        train_data = pd.concat(train_data).sample(frac=1., random_state=1312)
        val_data = pd.concat(val_data).sample(frac=1., random_state=1312)
        return dict(train_data), dict(val_data)

    @staticmethod
    def ohe_map(dataset):
        for key in dataset.keys():
            if key != "image":
                dataset[key] = tf.keras.backend.one_hot(indices=int(dataset[key]), num_classes=len(MAPPER[key]))
        dataset["image"] = tf.image.decode_image(tf.io.read_file(dataset["image"]), channels=3)
        labels = {"class": dataset.pop("class")}
        return dataset, labels

    def get_dataset(self, mode=None, repeat=None):
        if mode == "train":
            dataset = self.train_data
        elif mode == "val":
            dataset = self.val_data
        else:
            raise ValueError(f"{mode} is not a valid mode.")
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.map(lambda data: self.ohe_map(data), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_class_weights(self):
        return np.divide(len(self.train_data["class"]),
                         np.multiply(np.bincount(self.train_data["class"]),
                                     len(MAPPER["class"])))
