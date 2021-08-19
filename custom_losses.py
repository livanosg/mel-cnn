from itertools import product
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.keras.utils.losses_utils import ReductionV2


class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, mode, num_classes, from_logits=False, label_smoothing=0, reduction=ReductionV2.NONE,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.mode = mode
        self.num_classes = num_classes
        if self.mode == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 1.5, 1., 5., 20.]), axis=0)
        elif self.mode == 'ben_mal':
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 10.]), axis=0)
        else:
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 10.]), axis=0)

    def call(self, y_true, y_pred):
        weights = K.ones_like(y_pred) * self.weights
        return K.mean(K.sum(super().call(y_true, y_pred) * weights, axis=0))


class PerClassWeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, mode, num_classes, from_logits=False, label_smoothing=0, reduction=ReductionV2.AUTO,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.mode = mode
        self.num_classes = num_classes
        if self.mode == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            self.weights = tf.convert_to_tensor([[1., 1.5, 1., 5., 20.],
                                                 [10., 2., 2., 7., 10.],
                                                 [10., 5., 2., 8., 10.],
                                                 [10., 6., 5., 1., 10.],
                                                 [20., 6., 2., 4., 2.]])
        else:
            self.weights = tf.convert_to_tensor([[1., 10.],
                                                 [20., 1.]])

    def call(self, y_true, y_pred):
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), tf.float32)
        for c_p, c_t in product(range(self.num_classes), range(self.num_classes)):
            final_mask += (self.weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask
