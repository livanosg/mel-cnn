import tensorflow as tf


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        """ y_true: One-hot label
            y_pred: Softmax output."""
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        w_loss = y_true * tf.math.log(y_pred) * weights
        w_loss = -tf.reduce_sum(w_loss, -1)
        return w_loss

    return loss


def log_dice_loss(y_true, y_pred):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""
    with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
        class_freq = tf.reduce_sum(y_true, axis=[0, 1, 2])
        class_freq = tf.math.maximum(class_freq, 1)
        weights = 1 / (class_freq ** 2)
        numerator = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
        dice = (2 * weights * (numerator + 1)) / (weights * (denominator + 1))
    return tf.math.reduce_mean(- tf.math.log(dice))


def weighted_crossentropy(y_true, y_pred):  # todo fix weights shape
    class_freq = tf.reduce_sum(y_true, axis=[0, 1, 2], keepdims=True)
    class_freq = tf.math.maximum(class_freq, [1, 1])
    weights = tf.math.pow(tf.math.divide(tf.reduce_sum(class_freq), class_freq), 0.5)
    weights = tf.reduce_sum(y_true * weights, axis=-1)
    return tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred) * weights)


def custom_loss(y_true, y_pred):
    with tf.name_scope('Custom_loss'):
        dice_loss = log_dice_loss(y_true, y_pred)
        wce_loss = weighted_crossentropy(y_true, y_pred)
        loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, wce_loss)
    return loss
