import tensorflow as tf

from models import model_struct


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
            model = tf.keras.models.load_model(dirs['load_path'], compile=False)
        else:
            model = model_struct(args=args)
        if args['fine']:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False
    return model, strategy
