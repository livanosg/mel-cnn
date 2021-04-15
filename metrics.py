import tensorflow as tf
import tensorflow_addons as tfa


# --------------------------=========================== METRICS ==========================-----------------------------#

def metrics():
    # micro: True positives, false positives and false negatives are computed globally.
    # f1_micro = tfa.metrics.F1Score(num_classes=5, average='micro', name='f1_micro')
    # macro: True positives, false positives and false negatives are computed for each class
    # and their unweighted mean is returned.
    # f1_macro = tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_macro')
    # weighted: Metrics are computed for each class and returns the mean weighted by the
    # number of true instances in each class.
    f1_weighted = tfa.metrics.F1Score(num_classes=5, average='weighted', name='f1_weighted')
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=5)
    categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
    auc = tf.keras.metrics.AUC()
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    return [f1_weighted, mcc, auc, precision, recall]
