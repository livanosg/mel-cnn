import tensorflow as tf
import tensorflow_addons as tfa


def metrics():
    # macro: unweighted mean for each class
    f1_macro = tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_macro')
    auc = tf.keras.metrics.AUC()
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    cat_accuracy = tf.keras.metrics.CategoricalAccuracy()
    return [f1_macro, auc, precision, recall, cat_accuracy]
