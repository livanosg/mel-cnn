import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction


class WeightedCategoricalCrossentropy(CategoricalCrossentropy):
    def __init__(self, args, from_logits=False, weights=None, label_smoothing=0, reduction=Reduction.AUTO,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name="weighted_{}".format(name))
        self.task = args['task']
        self.num_classes = args['num_classes']
        if weights is not None:
            self.weights = tf.expand_dims(tf.convert_to_tensor(weights, dtype=tf.float32), axis=0)
        else:
            if self.task == '5cls':
                #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
                self.weights = tf.expand_dims(tf.convert_to_tensor([1., 3.811, 5.203, 1.33, 3.214], dtype=tf.float32), axis=0)
            elif self.task == 'ben_mal':
                self.weights = tf.expand_dims(tf.convert_to_tensor([1., 5.], dtype=tf.float32), axis=0)
            else:
                self.weights = tf.expand_dims(tf.convert_to_tensor([1., 5.], dtype=tf.float32), axis=0)

    def call(self, y_true, y_pred):
        mask = tf.reduce_sum(K.cast(K.equal(y_true, 1.), tf.float32) * self.weights, -1)
        return super().call(y_true, y_pred) * mask


class PerClassWeightedCategoricalCrossentropy(CategoricalCrossentropy):
    def __init__(self, args, from_logits=False, weights=None, label_smoothing=0, reduction=Reduction.AUTO,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, reduction, name=f"weighted_{name}")
        self.task = args['task']
        self.num_classes = args['num_classes']
        if self.task == '5cls':
            #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
            if weights is not None:
                self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)
            else:
                self.weights = tf.convert_to_tensor([[1., 1.21, 0.38, 1.286, 2.313],
                                                     [4.771, 3.811, 1.431, 1.362, 1.492],
                                                     [4.071, 2.321, 5.203, 5.161, 3.463],
                                                     [3.502, 0.734, 1.516, 1.33, 2.095],
                                                     [4.197, 0.670, 0.289, 0.913, 3.214]], dtype=tf.float32)
        elif self.task == 'ben_mal':
            self.weights = tf.convert_to_tensor(np.array([[1., 3.],
                                                          [5., 4.]]), dtype=tf.float32)
        else:
            self.weights = tf.convert_to_tensor(np.array([[1., 3.],
                                                          [5., 4.]]), dtype=tf.float32)

    def call(self, y_true, y_pred):
        weights = tf.gather_nd(self.weights, tf.concat([K.expand_dims(K.argmax(y_true, axis=-1)),
                                                        K.expand_dims(K.argmax(y_pred, axis=-1))], axis=-1))
        return super().call(y_true, y_pred) * weights


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed


def custom_loss(frac):
    def total_loss(y_true, y_pred):
        def log_dice_loss():
            """Inputs: y_pred: probs form per class
                       y_true: one-hot encoding of label
            """
            with tf.name_scope('Generalized_Dice_Log_Loss'):
                numerator = tf.reduce_sum(y_true * y_pred)
                denominator = tf.reduce_sum(y_true + y_pred)
                dice = tf.math.divide(x=2. * numerator, y=denominator)
            return - tf.math.log(dice)

        def weighted_crossentropy():
            """ y_true: One-hot label
                y_pred: Softmax output."""
            with tf.name_scope('Crossentropy_Loss'):
                wcce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(wcce)

        return tf.math.multiply(frac, log_dice_loss()) + tf.math.multiply(1. - frac, weighted_crossentropy())

    return total_loss
