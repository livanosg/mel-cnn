from tensorboard.plugins.hparams import api as hp

# --------------------------======================= HYPERPARAMETERS ======================-----------------------------#
LEARNING_RATE_RANGE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4, 1e-5]))
BATCH_SIZE_RANGE = hp.HParam('batch_size', hp.Discrete([4, 8, 16, 32, 64, 128, 256]))
HWC_RANGE = hp.HParam('hwc', hp.Discrete([224, 300, 256, 512]))
DROPOUT_RANGE = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
ACTIVATION_OPTIONS = hp.HParam('relu_a', hp.Discrete([0., 0.01, 0.05, 0.1, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_MODELS = hp.HParam('models', hp.Discrete(['xception', 'inception', 'efficientnet0', 'efficientnet1']))

# --------------------------=========================== METRICS ==========================-----------------------------#
METRIC_ACCURACY = 'accuracy'
