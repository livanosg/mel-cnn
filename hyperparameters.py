import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#

LEARNING_RATE_RANGE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4, 1e-5]))
BATCH_SIZE_RANGE = hp.HParam('batch_size', hp.Discrete([4, 8, 16, 32, 64, 128, 256]))
HWC_RANGE = hp.HParam('hwc', hp.Discrete([224, 256, 300, 512]))
DROPOUT_RANGE = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
ACTIVATION_OPTIONS = hp.HParam('relu_a', hp.Discrete([0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'adam']))
HP_MODELS = hp.HParam('models', hp.Discrete(['xception', 'inception', 'efficientnet0', 'efficientnet1']))


# --------------------------=========================== METRICS ==========================-----------------------------#

def metrics():
    METRIC_ACCURACY = 'accuracy'
    METRIC_CATEGORICAL_ACCURACY = 'categorical_accuracy'
    METRIC_AUC = 'AUC'
    METRIC_PRECISION = tf.keras.metrics.Precision(name='precision')
    METRIC_RECALL = tf.keras.metrics.Recall(name='recall')
    # METRIC_SPECIFICITY = ''
    return [METRIC_ACCURACY, METRIC_CATEGORICAL_ACCURACY, METRIC_AUC, METRIC_PRECISION, METRIC_RECALL]


# --------------------------=========================== ======= ==========================-----------------------------#

# test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
# activations = get_activations(model=custom_model, x=test)
# keract.display_activations(activations=activations, save=True, directory='test')
# keract.display_heatmaps(activations, test, save=True, directory='test')
