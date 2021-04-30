import tensorflow as tf
import numpy as np

import pandas as pd


class MelData:
    def __init__(self, file, frac=1., colour=None, hw=None, batch_size=None):
        self.features = pd.read_csv(file)
        self.targets = self.features.pop("class")
        self.features_dict = {}
        for column in self.features.columns:
            self.features_dict[column] = np.asarray(self.features[column])
        self.features_dict["image"] = f"proc_{str(hw)}_{colour}/" + self.features_dict["image"]
        self.target_dict = {"class": np.asarray(self.targets)}
        self.all_dict = self.features_dict, self.target_dict

        # THESE SHOULD BE PLACED IN PREPROCESSING LAYERS.
        self.image_type_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(num_oov_indices=0, vocabulary=np.unique(self.features_dict["image_type"]))
        self.image_type_onehot = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=self.image_type_lookup.vocab_size())
        self.target_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None, num_oov_indices=0, vocabulary=np.unique(self.target_dict["class"]))
        self.target_onehot = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=self.target_lookup.vocab_size())
        print(self.target_lookup.get_vocabulary())
        self.target_dict["class"] = self.target_onehot(self.target_lookup(self.target_dict["class"]))


a = MelData(file="all_data_init.csv", colour="rgb", hw=224)

print(a.target_dict)
