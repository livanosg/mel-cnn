import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding
import numpy as np

import pandas as pd


class MelData:
    def __init__(self, file, frac=1., colour=None, hw=None, batch=None):
        self.batch = batch
        self.features = pd.read_csv(file).astype(dtype=str)
        self.features = self.features.replace(np.nan, "", regex=True)
        self.features.pop("dataset_id")
        self.features["image"] = f"proc_{str(hw)}_{colour}/" + self.features["image"]

    def get_vocab_dict(self):
        vocab_dict = {}
        for column in self.features.columns:
            if column != "image":
                vocab_dict[column] = sorted(list(set(self.features[column])))
                if "nan" in vocab_dict[column]:
                    vocab_dict[column].remove("nan")
        return vocab_dict

    def onehot_encoding(self, column):
        vocab = self.get_vocab_dict()
        print(vocab[column])
        lookup = StringLookup(num_oov_indices=0, mask_token=None, vocabulary=vocab[column])

        lookup.adapt(np.asarray(self.features[column]))
        onehot = CategoryEncoding()
        onehot.adapt(np.asarray(lookup(lookup.get_vocabulary())))
        return onehot(lookup(self.features[column]))

    def get_data_dicts(self):
        features_dict = {}
        target_dict = {}
        for column in self.features.columns:
            if column == "class":
                target_dict["classes"] = self.onehot_encoding(column)
            else:
                if column != "image":
                    features_dict[column] = self.onehot_encoding(column)
        return features_dict, target_dict

    def get_dataset(self, repeat=None):
        dataset = tf.data.Dataset.from_tensor_slices(self.get_data_dicts())
        dataset = dataset.batch(self.batch)
        dataset = dataset.repeat(repeat)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


a = MelData(file="all_data_init.csv", colour="rgb", hw=224, batch=1)

data = a.get_dataset(1)

print(list(data.as_numpy_iterator())[2000])

for i in data.as_numpy_iterator():
    print(i)
    break


