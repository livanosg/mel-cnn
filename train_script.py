import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import models
from metrics import calc_metrics
from callbacks import LaterCheckpoint, EnrTensorboard
from custom_losses import binary_focal_loss, PerClassWeightedCategoricalCrossentropy, custom_loss


def setup_model(args):
    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('GPU')
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(args['load_model'], compile=False)
            for layer in model.layers:
                if layer.name.startswith('efficient') or layer.name.startswith('inception') or layer.name.startswith('xception'):
                    layer.trainable = False
            if args['fine'] and (not args['test']):
                model = unfreeze_model(model)
        else:
            model = models.model_struct(args=args)
    return model, strategy


def train_fn(args, data):
    loss_fn = {'cxe': 'categorical_crossentropy', 'focal': binary_focal_loss(), 'custom': custom_loss(args['loss_frac']),
               'perclass': PerClassWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]
    optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax, 'nadam': tf.keras.optimizers.Nadam,
                 'ftrl': tf.keras.optimizers.Ftrl, 'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                 'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]

    model, strategy = setup_model(args)
    lr = args['learning_rate'] * strategy.num_replicas_in_sync
    with strategy.scope():
        model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])

    with open(args['dir_dict']['model_summary'], 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    start_save = 0
    rlrop_patience = 10
    if args['fine']:
        rlrop_patience = 5
    es_patience = rlrop_patience * 2

    batch = args['batch_size'] * strategy.num_replicas_in_sync
    train_data = data.get_dataset(mode='train', batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
    val_data = data.get_dataset(mode='validation', batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=rlrop_patience),
                 tf.keras.callbacks.EarlyStopping(patience=es_patience),
                 tf.keras.callbacks.CSVLogger(filename=args['dir_dict']['train_logs'], separator=',', append=True),
                 LaterCheckpoint(filepath=args['dir_dict']['save_path'], save_best_only=True, start_at=start_save),
                 EnrTensorboard(val_data=val_data, log_dir=args['dir_dict']['logs'], class_names=args['class_names'])]
    model.fit(x=train_data, validation_data=val_data, steps_per_epoch=np.floor(data.train_len / batch),
              callbacks=callbacks, epochs=args['epochs'])
    tf.keras.backend.clear_session()


def val_fn(args, data):
    model, strategy = setup_model(args)
    val_data = data.get_dataset(mode='validation', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
    thresh_dist, thresh_f1 = calc_metrics(args=args, model=model, dataset=val_data, dataset_type='validation')
    if args['test']:
        # test_data = data.get_dataset(mode='test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
        # calc_metrics(args=args, model=model, dataset=test_data, dataset_type='test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
        if args['task'] == 'ben_mal':
            isic16_test_data = data.get_dataset(mode='isic16_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
            isic17_test_data = data.get_dataset(mode='isic17_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
            isic18_val_test_data = data.get_dataset(mode='isic18_val_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
            isic20_test_data = data.get_dataset(mode='isic20_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
            dermofit_test_data = data.get_dataset(mode='dermofit_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])
            up_test_data = data.get_dataset(mode='up_test', batch=64 * strategy.num_replicas_in_sync, no_image_type=args['no_image_type'], only_image=args['only_image'])

            calc_metrics(args=args, model=model, dataset=isic16_test_data, dataset_type='isic16_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
            calc_metrics(args=args, model=model, dataset=isic17_test_data, dataset_type='isic17_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
            calc_metrics(args=args, model=model, dataset=isic18_val_test_data, dataset_type='isic18_val_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
            calc_metrics(args=args, model=model, dataset=isic20_test_data, dataset_type='isic20_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
            calc_metrics(args=args, model=model, dataset=dermofit_test_data, dataset_type='dermofit_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)
            calc_metrics(args=args, model=model, dataset=up_test_data, dataset_type='up_test', thresh_dist=thresh_dist, thresh_f1=thresh_f1)


def unfreeze_model(trained_model):
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model
