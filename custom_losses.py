import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction


class WeightedCategoricalCrossentropy(CategoricalCrossentropy):
    def __init__(self, args, from_logits=False, label_smoothing=0, reduction=Reduction.AUTO,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.mode = args['mode']
        self.num_classes = args['num_classes']
        if self.mode == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 3.811, 5.203, 1.33, 3.214]), axis=0)
        elif self.mode == 'ben_mal':
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 5.]), axis=0)
        else:
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 5.]), axis=0)

    def call(self, y_true, y_pred):
        mask = tf.reduce_sum(K.cast(K.equal(y_true, 1.), tf.float32) * self.weights, -1)
        return super().call(y_true, y_pred) * mask


class PerClassWeightedCategoricalCrossentropy(CategoricalCrossentropy):
    def __init__(self, args, from_logits=False, label_smoothing=0, reduction=Reduction.AUTO,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.mode = args['mode']
        self.num_classes = args['num_classes']
        if self.mode == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            self.weights = tf.convert_to_tensor(np.array([[1., 1.21, 0.38, 1.286, 2.313],
                                                          [4.771, 3.811, 1.431, 1.362, 1.492],
                                                          [4.071, 2.321, 5.203, 5.161, 3.463],
                                                          [3.502, 0.734, 0.316, 1.33, 2.095],
                                                          [4.197, 0.670, 0.289, 0.913, 3.214]]), dtype=tf.float32)
        elif self.mode == 'ben_mal':
            self.weights = tf.convert_to_tensor(np.array([[1., 5.],
                                                          [2., 6.]]), dtype=tf.float32)
        else:
            self.weights = tf.convert_to_tensor(np.array([[1., 10.],
                                                          [4., 7.]]), dtype=tf.float32)

    def call(self, y_true, y_pred):
        weights = tf.gather_nd(self.weights, tf.concat([K.expand_dims(K.argmax(y_true, axis=-1)),
                                                        K.expand_dims(K.argmax(y_pred, axis=-1))], axis=-1))
        return super().call(y_true, y_pred) * weights
