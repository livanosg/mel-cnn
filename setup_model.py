import tensorflow as tf
from custom_losses import categorical_focal_loss
from custom_metrics import GeometricMean
from models_init import model_struct

tf.config.threading.set_inter_op_parallelism_threads(num_threads=16)
tf.config.threading.set_intra_op_parallelism_threads(num_threads=16)
tf.config.set_soft_device_placement(enabled=True)


def unfreeze_model(trained_model):
    """Make model trainable except BatchNormalization Layers"""
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model


def setup_model(args, dirs):
    """Setup training strategy. Select one of mirrored or singlegpu.
    Also check if a path to load a model is available and loads or setups a new model accordingly"""
    cross_device_ops = tf.distribute.HierarchicalCopyAllReduce() if args['os'] == 'win32'\
        else tf.distribute.NcclAllReduce()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops) if args['strategy'] == 'mirrored'\
        else tf.distribute.OneDeviceStrategy('GPU')
    assert args['gpus'] == strategy.num_replicas_in_sync
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(dirs['load_path'],
                                               compile=True,
                                               custom_objects={'categorical_focal_loss_fixed': categorical_focal_loss(),
                                                               'GeometricMean': GeometricMean})
        else:
            model = model_struct(args=args)
        if args['fine']:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False
    return model, strategy
