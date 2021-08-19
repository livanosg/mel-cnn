from itertools import product

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.keras.utils.losses_utils import ReductionV2


def custom_loss(weights):
    e = K.epsilon()
    weights = K.constant(weights)

    def total_loss(y_true, y_pred):
        def log_dice_loss(dice_y_true, dice_y_pred):
            """Inputs: y_pred: probs form per class
                       y_true: one-hot encoding of label
            """
            with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
                reduce_axis = list(range(len(dice_y_pred.shape)-1))
                numerator = tf.reduce_sum(dice_y_true * dice_y_pred, axis=reduce_axis)
                denominator = tf.reduce_sum(dice_y_true + dice_y_pred, axis=reduce_axis)
                dice = tf.math.divide(x=2. * numerator + e,
                                      y=denominator + e)
            return tf.reduce_mean(- tf.math.log(weights * dice))

        def weighted_crossentropy(wce_y_true, wce_y_pred):
            """ y_true: One-hot label
                y_pred: Softmax output."""
            with tf.name_scope('Weighted_Crossentropy_Loss'):
                wcce = K.categorical_crossentropy(wce_y_true, wce_y_pred) * K.sum(wce_y_true * weights, axis=-1)
                return tf.reduce_mean(wcce)
        return tf.math.multiply(.6, log_dice_loss(y_true, y_pred)) + tf.math.multiply(.4, weighted_crossentropy(y_true, y_pred))
    return total_loss


class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, mode, from_logits=False, label_smoothing=0, reduction=ReductionV2.SUM_OVER_BATCH_SIZE, name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.mode = mode
        if self.mode == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            self.weights = np.asarray([[1., 2.5, 2., 5., 20.],
                                       [10., 1., 2.5, 7., 8.],
                                       [10., 5., 1., 8., 6.],
                                       [10., 6., 5., 1., 5],
                                       [20., 6., 2., 4., 1.]])
        else:
            self.weights = np.asarray([[1., 5.],
                                       [5., 1.]])

    def call(self, y_true, y_pred):
        weights = self.weights
        num_classes = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(num_classes), range(num_classes)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask

