import tensorflow as tf
import tensorflow.keras.backend as K


def custom_loss(weights):
    e = tf.keras.backend.epsilon()
    Kweights = K.constant(weights)

    def total_loss(y_true, y_pred):
        def log_dice_loss(dice_y_true, dice_y_pred):
            """Inputs: y_pred: probs form per class
                       y_true: one-hot encoding of label
            """
            with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
                reduce_axis = list(range(len(dice_y_pred.shape)-1))
                numerator = tf.math.multiply(x=Kweights, y=tf.reduce_sum(dice_y_true * dice_y_pred, axis=reduce_axis))
                denominator = tf.math.multiply(x=Kweights, y=tf.reduce_sum(dice_y_true + dice_y_pred, axis=reduce_axis))
                with tf.name_scope('Dice_Division'):
                    division = tf.math.divide(x=tf.add(x=numerator, y=e),
                                              y=tf.add(x=denominator, y=e))
                    dice = tf.multiply(x=2., y=division)
                with tf.name_scope('Batch_loss'):
                    dice = - tf.math.log(dice)
                return tf.reduce_mean(dice, -1)

        def weighted_crossentropy(wce_y_true, wce_y_pred):
            """ y_true: One-hot label
                y_pred: Softmax output."""
            with tf.name_scope('Weighted_Crossentropy_Loss'):
                wcce = K.categorical_crossentropy(wce_y_true, wce_y_pred) * K.sum(wce_y_true * Kweights, axis=-1)
                return tf.reduce_mean(wcce, -1)
        return tf.math.multiply(.3, log_dice_loss(y_true, y_pred)) + tf.math.multiply(0.7, weighted_crossentropy(y_true, y_pred))
    return total_loss
