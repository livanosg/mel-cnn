import os
import tensorflow as tf
num_threads = os.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)
import tensorflow_addons as tfa
import models
from data_pipe import MelData
from metrics import calc_metrics
from callbacks import LaterCheckpoint, EnrTensorboard
from custom_losses import categorical_focal_loss, combined_loss, CMWeightedCategoricalCrossentropy


def unfreeze_model(trained_model):
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model


def setup_model(args):
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    # if args['strategy'] == 'mirrored':
    #     strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    # else:
    #     strategy = tf.distribute.OneDeviceStrategy('GPU')
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(args['load_model'], compile=False)
        else:
            model = models.model_struct(args=args)
        if args['fine']:
            model = unfreeze_model(model)
        else:
            for layer in model.layers:
                if layer.name.startswith('efficient') or layer.name.startswith('inception') or layer.name.startswith('xception'):
                    layer.trainable = False
    return model, strategy


def train_fn(args):
    model, strategy = setup_model(args)
    args['learning_rate'] = args['learning_rate'] * strategy.num_replicas_in_sync
    args['batch_size'] = args['batch_size'] * strategy.num_replicas_in_sync
    data = MelData(args=args)
    loss_fn = {'cxe': 'categorical_crossentropy', 'focal': categorical_focal_loss(alpha=[0.25, 0.25]),
               'combined': combined_loss(args['loss_frac']),
               'perclass': CMWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]

    def gmean_sens_spec(y_true, y_pred):
        y_pred_masked = tf.cast(tf.greater_equal(y_pred, tf.reduce_max(y_pred, axis=-1, keepdims=True)), dtype=tf.float32)
        tp = tf.cast(tf.math.equal(y_pred_masked[..., 1] * y_true[..., 1], 1.), dtype=tf.float32)
        tn = tf.cast(tf.math.equal(y_pred_masked[..., 0] * y_true[..., 0], 1.), dtype=tf.float32)
        fp = tf.cast(tf.math.equal(y_pred_masked[..., 1] * y_true[..., 0], 1.), dtype=tf.float32)
        fn = tf.cast(tf.math.equal(y_pred_masked[..., 0] * y_true[..., 1], 1.), dtype=tf.float32)
        sensitivity = tf.math.divide_no_nan(tf.reduce_sum(tp), (tf.reduce_sum(tp) + tf.reduce_sum(fn)))
        specificity = tf.math.divide_no_nan(tf.reduce_sum(tn), (tf.reduce_sum(fp) + tf.reduce_sum(tn)))
        g_mean = tf.math.sqrt((sensitivity * specificity))
        return g_mean

    with strategy.scope():
        optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax,
                     'nadam': tf.keras.optimizers.Nadam, 'ftrl': tf.keras.optimizers.Ftrl,
                     'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                     'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]
        model.compile(loss=loss_fn,
                      optimizer=optimizer(learning_rate=args['learning_rate']),
                      metrics=[gmean_sens_spec, tfa.metrics.F1Score(num_classes=args['num_classes'], average='weighted')])
                               # tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])

    with open(args['dir_dict']['model_summary'], 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    start_save = 10
    rop_patience = 2 * start_save
    if args['fine']:
        rop_patience = 5
    es_patience = 3 * rop_patience

    train_data = data.get_dataset(mode='train')
    val_data = data.get_dataset(mode='validation')
    callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch, lr: args['learning_rate'] * 10 if epoch < 5 else args['learning_rate']),  # warmup model
                 tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=rop_patience),
                 tf.keras.callbacks.EarlyStopping(patience=es_patience),
                 tf.keras.callbacks.CSVLogger(filename=args['dir_dict']['train_logs'], separator=',', append=True),
                 LaterCheckpoint(filepath=args['dir_dict']['save_path'], save_best_only=True, start_at=start_save, monitor='val_gmean_sens_spec', mode='max'),
                 EnrTensorboard(val_data=val_data, log_dir=args['dir_dict']['logs'], class_names=args['class_names'])]
    model.fit(x=train_data, validation_data=val_data, callbacks=callbacks, epochs=args['epochs'])  # steps_per_epoch=np.floor(data.train_len / batch),


def test_fn(args):
    model, strategy = setup_model(args)
    data = MelData(args=args)
    thresh_dist, thresh_f1 = calc_metrics(args=args, model=model, dataset=data.get_dataset(mode='validation',), dataset_type='validation')
    if args['image_type'] in ('both', 'derm'):
        for test in ('isic16_test', 'isic17_test', 'isic18_val_test', 'up_test', 'mclass_derm_test'):
            if args['task'] != 'nev_mel' and test != 'isic16_test':
                calc_metrics(args=args, model=model, dataset=data.get_dataset(mode=test), dataset_type=test, dist_thresh=thresh_dist, f1_thresh=thresh_f1)
    if args['image_type'] in ('both', 'clinic'):
        for test in ('dermofit_test', 'up_test', 'mclass_clinic_test'):
            calc_metrics(args=args, model=model, dataset=data.get_dataset(mode=test), dataset_type=test, dist_thresh=thresh_dist, f1_thresh=thresh_f1)
        if args['task'] == 'ben_mal':
            isic20_test_data = data.get_dataset(mode='isic20_test')
            calc_metrics(args=args, model=model, dataset=isic20_test_data, dataset_type='isic20_test', dist_thresh=thresh_dist, f1_thresh=thresh_f1)
