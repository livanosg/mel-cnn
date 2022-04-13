import os
import tensorflow as tf
import tensorflow_addons as tfa
import models
from features import TASK_CLASSES
from data_pipe import MelData
from metrics import calc_metrics
from callbacks import EnrTensorboard  # , LaterCheckpoint, LaterReduceLROnPlateau
from custom_losses import categorical_focal_loss, combined_loss, CMWeightedCategoricalCrossentropy


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
            assert os.path.exists(dirs['load_path'])
            model = tf.keras.models.load_model(dirs['load_path'], compile=False)
        else:
            model = models.model_struct(args=args)
        if args['fine']:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False
    return model, strategy


def train_fn(args, dirs):
    """Setup and run training stage"""
    model, strategy = setup_model(args, dirs)
    data = MelData(args, dirs)
    loss_fn = {'cxe': 'categorical_crossentropy',
               'focal': categorical_focal_loss(alpha=[1., 1.]),
               'combined': combined_loss(args['loss_frac']),
               'perclass': CMWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]

    def gmean(y_true, y_pred):
        y_pred_arg = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
        y_true_arg = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.float32)
        tp = tf.reduce_sum(y_true_arg * y_pred_arg)
        tn = tf.reduce_sum((1. - y_true_arg) * (1. - y_pred_arg))
        fp = tf.reduce_sum((1 - y_true_arg) * y_pred_arg)
        fn = tf.reduce_sum(y_true_arg * (1 - y_pred_arg))
        sensitivity = tf.math.divide_no_nan(tp, tp + fn)
        specificity = tf.math.divide_no_nan(tn, tn + fp)
        g_mean = tf.math.sqrt(sensitivity * specificity)
        return g_mean

    with strategy.scope():
        optimizer = {'adam': tf.keras.optimizers.Adam,
                     'adamax': tf.keras.optimizers.Adamax,
                     'nadam': tf.keras.optimizers.Nadam,
                     'ftrl': tf.keras.optimizers.Ftrl,
                     'rmsprop': tf.keras.optimizers.RMSprop,
                     'sgd': tf.keras.optimizers.SGD,
                     'adagrad': tf.keras.optimizers.Adagrad,
                     'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]
        model.compile(loss=loss_fn, optimizer=optimizer(learning_rate=args['learning_rate'] * args['gpus']),
                      metrics=[tfa.metrics.F1Score(num_classes=len(TASK_CLASSES[args['task']]),
                                                   average='macro', name='f1_macro'),
                               gmean])

    with open(dirs['model_summary'], 'w', encoding='utf-8') as model_summary:
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))

    es_patience = 30

    train_data = data.get_dataset(dataset_name='train')
    val_data = data.get_dataset(dataset_name='validation')
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=es_patience, verbose=1, monitor='val_gmean',
                                                  mode='max', restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'],
                                              separator=',', append=True),
                 EnrTensorboard(val_data=val_data, log_dir=dirs['logs'],
                                class_names=TASK_CLASSES[args['task']])]
    model.fit(x=train_data, validation_data=val_data, callbacks=callbacks,
              epochs=args['epochs'])  # steps_per_epoch=np.floor(data.train_len / batch),
    model.save(filepath=dirs['save_path'])


def test_fn(args, dirs):
    """ run validation and tests"""
    if args['image_type'] not in ('clinic', 'derm'):
        raise ValueError(f'{args["image_type"]} not valid. Select on one of ("clinic", "derm")')
    model, _ = setup_model(args, dirs)
    data = MelData(args, dirs)
    data.args['clinic_val'] = False
    thr_d, thr_f1 = calc_metrics(args=args, dirs=dirs, model=model,
                                 dataset=data.get_dataset(dataset_name='validation'),
                                 dataset_name='validation')
    test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                              'mclass_derm_test', 'up_test'],
                     'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
    if args['task'] == 'nev_mel':
        test_datasets['derm'].remove('isic16_test')

    for test_dataset in test_datasets[args['image_type']]:
        calc_metrics(args=args, dirs=dirs, model=model,
                     dataset=data.get_dataset(dataset_name=test_dataset),
                     dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
        if args['task'] == 'ben_mal':
            calc_metrics(args=args, dirs=dirs, model=model,
                         dataset=data.get_dataset(dataset_name='isic20_test'),
                         dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)
