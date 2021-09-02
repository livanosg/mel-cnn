import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorboard.plugins.hparams import api as hp

from custom_losses import binary_focal_loss, PerClassWeightedCategoricalCrossentropy, custom_loss
from data_pipe import MelData
from metrics import calc_metrics
from model import model_fn
from callbacks import LaterCheckpoint, EnrTensorboard, TestCallback


def train_val_test(args):
    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('GPU')
    args['replicas'] = strategy.num_replicas_in_sync

    if not args['test'] and not args['validate']:
        os.makedirs(args['dir_dict']['logs'], exist_ok=True)
        os.makedirs(args['dir_dict']['trial'], exist_ok=True)

    data = MelData(args=args)
    args['train_data'] = data.get_dataset(mode='train', batch=args['batch_size'] * args['replicas'])
    args['val_data'] = data.get_dataset(mode='val', batch=args['batch_size'] * args['replicas'])
    args['test_data'] = data.get_dataset(mode='test', batch=args['batch_size'] * args['replicas'])
    args['isic20_test'] = data.get_dataset(mode='isic20_test', batch=args['batch_size'] * args['replicas'])
    if args['test'] or args['validate']:
        args['dir_dict']['model_path'] = args['test_model']
        args['dir_dict']['trial'] = os.path.dirname(args['dir_dict']['model_path'])
        model = tf.keras.models.load_model(args['dir_dict']['model_path'], compile=False)
        if args['test']:
            if args['task'] in ('ben_mal', '5cls'):
                calc_metrics(args=args, model=model, dataset=args['isic20_test'], dataset_type='isic20_test')
        else:
            calc_metrics(args=args, model=model, dataset=args['val_data'], dataset_type='validation')
            calc_metrics(args=args, model=model, dataset=args['test_data'], dataset_type='test')
            if args['task'] in ('ben_mal', '5cls'):
                calc_metrics(args=args, model=model, dataset=args['isic20_test'], dataset_type='isic20_test')
    else:
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            [f.write(': '.join([key.capitalize().rjust(len(max(args.keys(), key=len))), str(args[key])]) + '\n')
             for key in args.keys() if key not in ('dir_dict', 'hparams', 'train_data', 'val_data', 'test_data', 'isic20_test')]
        with strategy.scope():
            loss_fn = {'cxe': 'categorical_crossentropy',
                       'focal': binary_focal_loss(),
                       'custom': custom_loss(args['loss_frac']),
                       'perclass': PerClassWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]
            optimizer = {'adam': tf.keras.optimizers.Adam, 'ftrl': tf.keras.optimizers.Ftrl,
                         'sgd': tf.keras.optimizers.SGD, 'rmsprop': tf.keras.optimizers.RMSprop,
                         'adadelta': tf.keras.optimizers.Adadelta, 'adagrad': tf.keras.optimizers.Adagrad,
                         'adamax': tf.keras.optimizers.Adamax, 'nadam': tf.keras.optimizers.Nadam}[args['optimizer']]
            custom_model = model_fn(args=args)
            with open(os.path.join(args['dir_dict']['trial'], 'model_summary.txt'), 'w') as f:
                custom_model.summary(print_fn=lambda x: f.write(x + '\n'))
            custom_model.compile(optimizer=optimizer(learning_rate=args['learning_rate'] * args['replicas']), loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])
        # --------------------------------------------------- Callbacks ---------------------------------------------- #

        callbacks = [ReduceLROnPlateau(factor=0.1, patience=10, verbose=args['verbose']),
                     EarlyStopping(patience=20, verbose=args['verbose']),
                     LaterCheckpoint(filepath=args['dir_dict']['model_path'], monitor='val_f1_score', mode='max', save_best_only=True, start_at=10, verbose=args['verbose']),
                     EnrTensorboard(log_dir=args['dir_dict']['logs'], val_data=args['val_data'], class_names=args['class_names']),
                     hp.KerasCallback(args['dir_dict']['logs'], hparams={'pretrained': args['pretrained'], 'task':args['task'],
                                                                         'image_type':args['image_type'], 'image_size':args['image_size'],
                                                                         'only_image':args['only_image'], 'colour':args['colour'],
                                                                         'batch_size':args['batch_size'], 'learning_rate':args['learning_rate'],
                                                                         'optimizer':args['optimizer'], 'activation':args['activation'],
                                                                         'dropout':args['dropout'], 'epochs':args['epochs'],
                                                                         'conv_layers':args['conv_layers'], 'no_image_weights':args['no_image_weights'],
                                                                         'no_image_type':args['no_image_type']},
                                      trial_id=os.path.basename(args["dir_dict"]["trial"])),
                     TestCallback(args=args)]

        train_1 = custom_model.fit(x=args['train_data'], validation_data=args['val_data'], epochs=args['epochs'], callbacks=callbacks, verbose=args['verbose'])


# -------------------------------- FINE TUNING ------------------------------------------------------------------------ #
        n_epochs = len(train_1.history['loss'])
        args['learning_rate'] = 1e-6
        args['batch_size'] = 4
        args['train_data'] = data.get_dataset(mode='train', batch=args['batch_size'] * args['replicas'])
        args['val_data'] = data.get_dataset(mode='val', batch=args['batch_size'] * args['replicas'])
        args['test_data'] = data.get_dataset(mode='test', batch=args['batch_size'] * args['replicas'])
        args['isic20_test'] = data.get_dataset(mode='isic20_test', batch=args['batch_size'] * args['replicas'])

        def unfreeze_model(trained_model):
            for layer in trained_model.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            return trained_model

        with strategy.scope():
            custom_model = unfreeze_model(custom_model)
            custom_model.compile(optimizer=optimizer(learning_rate=args['learning_rate'] * args['replicas']), loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])  # binary_focal_loss()

        with open(os.path.join(args['dir_dict']['trial'], 'fine_model_summary.txt'), 'w') as f:
            custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

        args['dir_dict']['trial'] = os.path.join(args['dir_dict']['trial'], 'fine')
        args['dir_dict']['logs'] = os.path.join(args['dir_dict']['logs'], 'fine')
        args['dir_dict']['model_path'] = os.path.join(args['dir_dict']['model_path'], 'fine')

        callbacks = [ReduceLROnPlateau(factor=0.1, patience=10, verbose=args['verbose']),
                     EarlyStopping(patience=15, verbose=args['verbose']),
                     LaterCheckpoint(filepath=args['dir_dict']['model_path'], monitor='val_f1_score', mode='max', save_best_only=True, start_at=n_epochs + 0, verbose=args['verbose']),
                     EnrTensorboard(log_dir=args['dir_dict']['logs'], val_data=args['val_data'], class_names=args['class_names']),
                     hp.KerasCallback(args['dir_dict']['logs'],
                                      hparams={'pretrained': args['pretrained'], 'task':args['task'],
                                               'image_type':args['image_type'], 'image_size':args['image_size'],
                                               'only_image':args['only_image'], 'colour':args['colour'],
                                               'batch_size':args['batch_size'], 'learning_rate':args['learning_rate'],
                                               'optimizer':args['optimizer'], 'activation':args['activation'],
                                               'dropout':args['dropout'], 'epochs':args['epochs'],
                                               'conv_layers':args['conv_layers'], 'no_image_weights':args['no_image_weights'],
                                               'no_image_type':args['no_image_type']},
                                      trial_id=args["dir_dict"]["trial"].split('/')[-2] + '_fine'),
                     TestCallback(args=args)]
        custom_model.fit(x=args['train_data'], validation_data=args['val_data'], initial_epoch=n_epochs, epochs=n_epochs + 10, callbacks=callbacks, verbose=args['verbose'])
