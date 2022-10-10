import tensorflow as tf


class CMWeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, args, from_logits=False, weights=None, label_smoothing=0,
                 name='categorical_crossentropy'):
        super().__init__(from_logits, label_smoothing, name=f"weighted_{name}")
        self.task = args['task']
        if weights is not None:
            self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            if self.task == '5cls':
                #  0: Nevus | 1: Non-Nevus benign | 2: Suspicious | 3: Non-Melanocytic Carcinoma | 4: Melanoma
                self.weights = tf.constant([[1., 1.21, 0.38, 1.286, 2.313],
                                            [4.771, 3.811, 1.431, 1.362, 1.492],
                                            [4.071, 2.321, 5.203, 5.161, 3.463],
                                            [3.502, 0.734, 1.516, 1.33, 2.095],
                                            [4.197, 0.670, 0.289, 0.913, 3.214]], dtype=tf.float32)
            if self.task == 'ben_mal':  # 0: Benign | 1: Malignant
                self.weights = tf.constant([[1., 3.],
                                            [5., 4.]], dtype=tf.float32)
            if self.task == 'nev_mel':  # 0: Nevus | 1: Melanoma
                self.weights = tf.constant([[1., 3.],
                                            [5., 4.]], dtype=tf.float32)

    def call(self, y_true, y_pred):
        weights = tf.gather_nd(self.weights, tf.concat([tf.expand_dims(tf.math.argmax(y_true, axis=-1)),
                                                        tf.expand_dims(tf.math.argmax(y_pred, axis=-1))], axis=-1))
        return super().call(y_true, y_pred) * weights


def categorical_focal_loss(alpha=[1., 1.], gamma=2.):
    # categorical focal loss from https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    # @tf.keras.utils.register_keras_serializable()
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = - tf.math.multiply(x=y_true, y=tf.math.log(y_pred))
        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return categorical_focal_loss_fixed


def combined_loss(frac, gamma=2.):
    def dice_loss(y_true, y_pred):
        with tf.name_scope('Generalized_Dice_Log_Loss'):
            numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
            denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
            dice = tf.clip_by_value(2 * numerator / denominator, 1e-7, 1 - 1e-7)
        return - tf.math.log(dice)

    def cxe(y_true, y_pred):
        a = 1.
        with tf.name_scope('Crossentropy_Loss'):
            focal_param = tf.reduce_sum(a * tf.math.pow(1 - y_pred, gamma) * y_true, axis=-1)
            return focal_param * tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)

    def total_loss(y_true, y_pred):
        """Inputs: y_pred: probs form per class
                   y_true: one-hot encoding of label
        """
        if frac == 0.:
            return cxe(y_true, y_pred)
        elif frac == 1.:
            return dice_loss(y_true, y_pred)
        else:
            return tf.add(x=tf.math.multiply(x=frac, y=dice_loss(y_true, y_pred)),
                          y=tf.math.multiply(x=1. - frac, y=cxe(y_true, y_pred)))

    return total_loss


def losses(args):
    return {'cxe': 'categorical_crossentropy',
            'focal': categorical_focal_loss(alpha=[1., 1.]),
            'combined': combined_loss(args['loss_frac']),
            'perclass': CMWeightedCategoricalCrossentropy(args=args)}
