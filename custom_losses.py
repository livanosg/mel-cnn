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
            self.weights = tf.expand_dims(tf.convert_to_tensor([1., 1.5, 1., 5., 20.]), axis=0)
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
            self.weights = tf.convert_to_tensor([[1., 1., 1., 2., 2.],
                                                 [3., 6., 1., 2., 3.],
                                                 [1., 1., 7., 4., 2.],
                                                 [5., 1., 1., 3., 2.],
                                                 [7., 1., 1., 2., 3.]])
            # [1., 1.5, 1., 5., 20.]
            # [10., 3., 2., 7., 10.]
            # [10., 5., 3., 8., 10.]
            # [10., 6., 5., 3., 10.]
            # [20., 6., 2., 4., 5.]]
            #
            #

        elif self.mode == 'ben_mal':
            self.weights = tf.convert_to_tensor([[1., 5.],
                                                 [10., 5.]])
        else:
            self.weights = tf.convert_to_tensor([[1., 4.],
                                                 [10., 5.]])

    def call(self, y_true, y_pred):
        weights = tf.gather_nd(self.weights, tf.concat([K.expand_dims(K.argmax(y_true, axis=-1)),
                                                        K.expand_dims(K.argmax(y_pred, axis=-1))], axis=-1))
        return super().call(y_true, y_pred) * weights
