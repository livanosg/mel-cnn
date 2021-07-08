import tensorflow as tf


def weighted_categorical_crossentropy(reg_loss, weights):
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
        return w_loss + reg_loss

    return loss
