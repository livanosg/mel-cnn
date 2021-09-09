import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import models
from data_pipe import MelData
from metrics import calc_metrics
from callbacks import LaterCheckpoint, EnrTensorboard
from custom_losses import binary_focal_loss, PerClassWeightedCategoricalCrossentropy, custom_loss


def train_val_test(args):
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #         tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*12)])
    #     except RuntimeError as e:
    #         print(e)  # Virtual devices must be set before GPUs have been initialized
        # try:  # Currently, memory growth needs to be the same across GPUs
        #     [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        # except RuntimeError as e:
        #     print(e)  # Memory growth must be set before GPUs have been initialized

    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('GPU')
    batch = args['batch_size'] * strategy.num_replicas_in_sync
    lr = args['learning_rate'] * strategy.num_replicas_in_sync

    if not args['test']:
        os.makedirs(args['dir_dict']['trial'], exist_ok=True)
        os.makedirs(args['dir_dict']['logs'], exist_ok=True)

        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', os.path.basename(args['dir_dict']['trial'])])
            [writer.writerow([key, str(args[key])]) for key in args.keys() if key != 'dir_dict']

    data = MelData(task=args['task'], image_type=args['image_type'], pretrained=args['pretrained'],
                   dir_dict=args['dir_dict'], input_shape=args['input_shape'], dataset_frac=args['dataset_frac'])
    loss_fn = {'cxe': 'categorical_crossentropy', 'focal': binary_focal_loss(), 'custom': custom_loss(args['loss_frac']),
               'perclass': PerClassWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]
    optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax, 'nadam': tf.keras.optimizers.Nadam,
                 'ftrl': tf.keras.optimizers.Ftrl, 'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                 'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]
    with strategy.scope():
        if args['load_model']:
            model = tf.keras.models.load_model(args['load_model'], compile=False)
        else:
            model = models.model_struct(args=args)
    start_save = 10
    init_epoch = 0
    if (not args['fine']) and (not args['test']):
        with open(os.path.join(args['dir_dict']['trial'], 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        data.logs()
        with strategy.scope():
            model.compile(optimizer=optimizer(learning_rate=lr), loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])
        all_data = data.all_datasets(batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
        callbacks = callback_list(args, val_data=all_data['validation'], start_save=start_save)
        train_results = model.fit(x=all_data['train'], validation_data=all_data['validation'], steps_per_epoch=np.floor(data.train_len / batch),
                                  callbacks=callbacks, epochs=args['epochs'], verbose=args['verbose'])
        init_epoch = len(train_results.history['loss'])
        args['fine'] = True
        lr = 1e-6 * strategy.num_replicas_in_sync
        batch = 4 * strategy.num_replicas_in_sync
    if args['fine'] and (not args['test']):
        if not args['load_model']:
            args['dir_dict']['trial'] = os.path.join(args['dir_dict']['trial'], 'fine')
            args['dir_dict']['logs'] = os.path.join(args['dir_dict']['logs'], 'fine')
            args['dir_dict']['model_path'] = os.path.join(args['dir_dict']['trial'], 'model')
            os.makedirs(args['dir_dict']['trial'], exist_ok=True)
            os.makedirs(args['dir_dict']['logs'], exist_ok=True)
        with strategy.scope():
            model_fine = unfreeze_model(model)
            model_fine.compile(optimizer=optimizer(learning_rate=lr), loss=loss_fn,
                               metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])
        all_data = data.all_datasets(batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
        callbacks = callback_list(args, val_data=all_data['validation'], start_save=start_save)
        model_fine.fit(x=all_data['train'], validation_data=all_data['validation'], callbacks=callbacks,
                       initial_epoch=init_epoch, epochs=init_epoch + args['epochs'], steps_per_epoch=np.floor(data.train_len / batch), verbose=args['verbose'])
        args['test'] = True
    if args['test']:
        batch = 64
        if args['fine']:
            batch = 10
        all_data = data.all_datasets(batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
        calc_metrics(args=args, model=model, dataset=all_data['validation'], dataset_type='validation')
        calc_metrics(args=args, model=model, dataset=all_data['test'], dataset_type='test')
        if args['task'] == 'ben_mal':
            calc_metrics(args=args, model=model, dataset=all_data['isic20_test'], dataset_type='isic20_test')


def unfreeze_model(trained_model):
    for layer in trained_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return trained_model


def callback_list(args, val_data, start_save=10):
    rop_patience = 10
    es_patience = rop_patience * 2
    if args['fine']:
        rop_patience = 5
        es_patience = rop_patience * 2
    return [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=rop_patience, verbose=args['verbose']),
            tf.keras.callbacks.EarlyStopping(patience=es_patience, verbose=args['verbose']),
            LaterCheckpoint(filepath=args['dir_dict']['model_path'], save_best_only=True, start_at=start_save, verbose=args['verbose']),
            EnrTensorboard(val_data=val_data, log_dir=args['dir_dict']['logs'], class_names=args['class_names'])]
