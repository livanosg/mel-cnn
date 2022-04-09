import tensorflow as tf
import tensorflow_addons as tfa
import models
from data_pipe import MelData
from metrics import calc_metrics
from callbacks import LaterCheckpoint, EnrTensorboard, LaterReduceLROnPlateau
from custom_losses import categorical_focal_loss, combined_loss, CMWeightedCategoricalCrossentropy


def unfreeze_model(trained_model):
    """Make model trainable except BatchNormalization Layers"""
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model


def setup_model(args):
    """Setup training strategy. Select one of mirrored or singlegpu.
    Also check if a path to load a model is available and loads or setups a new model accordingly"""
    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('GPU')
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(args['load_model'], compile=False)
        else:
            model = models.model_struct(args=args)
        if args['fine']:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith(('efficient', 'inception', 'xception')):
                    layer.trainable = False
    return model, strategy


def train_fn(args):
    """Setup and run training stage"""
    model, strategy = setup_model(args)
    args['learning_rate'] = args['learning_rate'] * strategy.num_replicas_in_sync
    args['batch_size'] = args['batch_size'] * strategy.num_replicas_in_sync
    data = MelData(args=args)
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
        model.compile(loss=loss_fn,
                      optimizer=optimizer(learning_rate=args['learning_rate']),
                      metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'],
                                                   average='macro', name='f1_macro'),
                               gmean])

    with open(args['dir_dict']['model_summary'], 'w', encoding='utf-8') as model_summary:
        model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    # start_save = 10
    # rop_patience = 5
    # if args['fine']:
        # rop_patience = 5
    es_patience = 30

    train_data = data.get_dataset(dataset_name='train')
    val_data = data.get_dataset(dataset_name='validation')

    # def schedule(epoch, learning_rate):
    #     if epoch < 0:
    #         new_lr = args['learning_rate'] * 10
    #     else:
    #         new_lr = args['learning_rate']
    #     return new_lr
    # tf.keras.callbacks.LearningRateScheduler(schedule=schedule),
    # LaterReduceLROnPlateau(factor=0.1, patience=rop_patience,
    #                                         monitor="val_gmean", mode='max', start_at=rop_patience, verbose=1),
    # LaterCheckpoint(filepath=args['dir_dict']['save_path'],
    #                 save_best_only=True, start_at=start_save,
    #                 monitor='val_gmean', mode='max'),

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=es_patience, verbose=1, monitor='val_gmean', mode='max', restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(filename=args['dir_dict']['train_logs'],
                                              separator=',', append=True),
                 EnrTensorboard(val_data=val_data, log_dir=args['dir_dict']['logs'],
                                class_names=args['class_names'])]
    model.fit(x=train_data, validation_data=val_data, callbacks=callbacks,
              epochs=args['epochs'])  # steps_per_epoch=np.floor(data.train_len / batch),
    model.save(filepath=args['dir_dict']['save_path'])


def test_fn(args):
    """ run validation and tests"""
    if args['image_type'] not in ('clinic', 'derm'):
        raise ValueError(f'{args["image_type"]} not valid. Select on one of ("clinic", "derm")')
    model, _ = setup_model(args)
    data = MelData(args=args)
    thr_d, thr_f1 = calc_metrics(args=args, model=model,
                                 dataset=data.get_dataset(dataset_name='validation'),
                                 dataset_name='validation')
    test_datasets = {'derm': ['isic16_test', 'isic17_test', 'isic18_val_test',
                              'mclass_derm_test', 'up_test'],
                     'clinic': ['up_test', 'dermofit_test', 'mclass_clinic_test']}
    if args['task'] == 'nev_mel':
        test_datasets['derm'].remove('isic16_test')

    for test_dataset in test_datasets[args['image_type']]:
        calc_metrics(args=args, model=model,
                     dataset=data.get_dataset(dataset_name=test_dataset),
                     dataset_name=test_dataset, dist_thresh=thr_d, f1_thresh=thr_f1)
        if args['task'] == 'ben_mal':
            calc_metrics(args=args, model=model,
                         dataset=data.get_dataset(dataset_name='isic20_test'),
                         dataset_name='isic20_test', dist_thresh=thr_d, f1_thresh=thr_f1)
