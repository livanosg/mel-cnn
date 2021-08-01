import tensorflow as tf
import tensorflow.keras.backend as K


def custom_loss(weights):
    e = tf.keras.backend.epsilon()
    weights = K.constant(weights)

    def total_loss(y_true, y_pred):
        def log_dice_loss(dice_y_true, dice_y_pred):
            """Inputs: y_pred: probs form per class
                       y_true: one-hot encoding of label
            """
            with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
                reduce_axis = list(range(len(dice_y_pred.shape)-1))
                numerator = weights * tf.reduce_sum(dice_y_true * dice_y_pred, axis=reduce_axis)
                denominator = weights * tf.reduce_sum(dice_y_true + dice_y_pred, axis=reduce_axis)
                dice = tf.math.divide(x=2. * tf.reduce_sum(numerator),
                                      y=tf.reduce_sum(denominator))
            return - tf.math.log(dice)

        def weighted_crossentropy(wce_y_true, wce_y_pred):
            """ y_true: One-hot label
                y_pred: Softmax output."""
            with tf.name_scope('Weighted_Crossentropy_Loss'):
                wcce = K.categorical_crossentropy(wce_y_true, wce_y_pred) * K.sum(wce_y_true * weights, axis=-1)
                return tf.reduce_mean(wcce)
        return tf.math.multiply(.6, log_dice_loss(y_true, y_pred)) + tf.math.multiply(.4, weighted_crossentropy(y_true, y_pred))
    return total_loss
