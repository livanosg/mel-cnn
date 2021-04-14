import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_addons as tfa
# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#

LR_LST = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))  # , 1e-5]))
BATCH_SIZE_RANGE = hp.HParam('batch_size', hp.Discrete([256]))
HWC_DOM = hp.HParam('hwc', hp.Discrete([224, 256]))  # , 300, 512]))
DROPOUT_LST = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
RELU_A = hp.HParam('relu_a', hp.Discrete([0.]))  # , 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]))
OPTIMIZER_LST = hp.HParam('optimizer', hp.Discrete(['adam']))  # 'ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam',
MODEL_LST = hp.HParam('models', hp.Discrete(['efficientnet0']))  # ['xception', 'inception', 'efficientnet1']))


# --------------------------=========================== METRICS ==========================-----------------------------#

def metrics():
    # micro: True positivies, false positives and false negatives are computed globally.
    # f1_micro = tfa.metrics.F1Score(num_classes=5, average='micro', name='f1_micro')
    # macro: True positivies, false positives and false negatives are computed for each class
    # and their unweighted mean is returned.
    # f1_macro = tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_macro')
    # weighted: Metrics are computed for each class and returns the mean weighted by the
    # number of true instances in each class.
    f1_weighted = tfa.metrics.F1Score(num_classes=5, average='weighted', name='f1_weighted')
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=5)
    # accuracy = 'accuracy'
    # categorical_accuracy = 'categorical_accuracy'
    auc = 'AUC'
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    return [f1_weighted, mcc, auc, precision, recall]


# --------------------------=========================== ======= ==========================-----------------------------#

# test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
# activations = get_activations(model=custom_model, x=test)
# keract.display_activations(activations=activations, save=True, directory='test')
# keract.display_heatmaps(activations, test, save=True, directory='test')

# keras.utils.plot_model(custom_model, 'custom_model.png', rankdir='LR', show_layer_names=False)
