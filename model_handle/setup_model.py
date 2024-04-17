import tensorflow as tf
# from tensorflow.distribute import HierarchicalCopyAllReduce, NcclAllReduce, MirroredStrategy, OneDeviceStrategy
from tensorflow.python.distribute.strategy_combinations import MirroredStrategy, OneDeviceStrategy
from train_handle.custom_losses import categorical_focal_loss
from train_handle.custom_metrics import GeometricMean
from model_handle.models_init import model_struct

tf.config.threading.set_inter_op_parallelism_threads(num_threads=16)
tf.config.threading.set_intra_op_parallelism_threads(num_threads=16)
tf.config.set_soft_device_placement(enabled=True)


def unfreeze_model(trained_model):
    """Make model trainable except BatchNormalization Layers"""
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model


def setup_model(args, strategy, load_path=None, finetune=False):
    """Setup training strategy. Select one of mirrored or singlegpu.
    Also check if a path to load a model is available and loads or setups a new model accordingly"""
    # cross_device_ops = HierarchicalCopyAllReduce() if sys.platform == 'win32' else NcclAllReduce()
    # strategy = MirroredStrategy(cross_device_ops=cross_device_ops) if strategy == 'mirrored' else OneDeviceStrategy('GPU')
    strategy = MirroredStrategy() if strategy == 'mirrored' else OneDeviceStrategy('GPU')
    with strategy.scope():
        if load_path:
            model = tf.keras.models.load_model(load_path, compile=True,
                                               custom_objects={'categorical_focal_loss_fixed': categorical_focal_loss(),
                                                               'GeometricMean': GeometricMean})
        else:
            model = model_struct(args=args)
        if finetune:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False
    return model, strategy
