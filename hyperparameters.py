import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#

LR_LST = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4, 1e-5]))
BATCH_SIZE_RANGE = hp.HParam('batch_size', hp.Discrete([4, 8, 16, 32, 64, 128, 256]))
HWC_RNG = hp.HParam('hwc', hp.Discrete([224, 256, 300, 512]))
DROPOUT_LST = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
RELU_A_LST = hp.HParam('relu_a', hp.Discrete([0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]))
OPTIMIZER_LST = hp.HParam('optimizer', hp.Discrete(['ftrl', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'adam']))
MODEL_LST = hp.HParam('models', hp.Discrete(['xception', 'inception', 'efficientnet0', 'efficientnet1']))


# --------------------------=========================== METRICS ==========================-----------------------------#

def metrics():
    metric_accuracy = 'accuracy'
    metric_categorical_accuracy = 'categorical_accuracy'
    metric_auc = 'AUC'
    metric_precision = tf.keras.metrics.Precision(name='precision')
    metric_recall = tf.keras.metrics.Recall(name='recall')
    return [metric_accuracy, metric_categorical_accuracy, metric_auc, metric_precision, metric_recall]


# --------------------------=========================== ======= ==========================-----------------------------#

# test = np.asarray(tf.expand_dims(eval_data.as_numpy_iterator().next()[0]['image'][0], axis=0))
# activations = get_activations(model=custom_model, x=test)
# keract.display_activations(activations=activations, save=True, directory='test')
# keract.display_heatmaps(activations, test, save=True, directory='test')

# keras.utils.plot_model(custom_model, 'custom_model.png', rankdir='LR', show_layer_names=False)
