from config import MAPPER, directories
import tensorflow as tf
import numpy as np
import pandas as pd


class MelData:
    def __init__(self, file: str, frac: float = 1., img_folder: str = None, batch: int = None):
        self.batch = batch
        self.frac = frac
        self.features = pd.read_csv(file, dtype="str")
        self.features.pop("dataset_id")
        self.features.fillna(-10, inplace=True)
        self.features.replace(to_replace=MAPPER, inplace=True)
        self.features["image"] = img_folder + self.features["image"]
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
        return dict(train_data), dict(val_data)

    @staticmethod
    def ohe_map(features):
        """ Turn features to one-hot encoded vectors. Also return a pair of (features, labels) dicts.
        Inputs:
        features: dictionary of features int encoded."""
        for key in features.keys():
            if key != "image":
                features[key] = tf.keras.backend.one_hot(indices=np.asarray(features[key]),
                                                         num_classes=len(MAPPER[key]))
        labels = {"class": features.pop("class")}
        return features, labels

    @staticmethod
    def imread(features, labels):
        """Read image from (features, labels) dicts and return"""
        features["image"] = tf.image.decode_image(tf.io.read_file(features["image"]), channels=3)
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
                         np.multiply(np.sum(self.train_data[1]["class"], axis=0), len(MAPPER["class"])))


if __name__ == '__main__':
    img_folder = directories(run_num=0, img_size=224, colour="rgb")["image_folder"]
    a = MelData("all_data_init.csv", frac=0.1, img_folder=img_folder, batch=1)
    a = a.get_dataset(mode="train", repeat=1)
    for i in a.as_numpy_iterator():
        print(i)
        break
