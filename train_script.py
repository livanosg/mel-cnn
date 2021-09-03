import csv
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorboard.plugins.hparams import api as hp

from custom_losses import binary_focal_loss, PerClassWeightedCategoricalCrossentropy, custom_loss
from data_pipe import MelData
from metrics import calc_metrics
from model import model_struct
from callbacks import LaterCheckpoint, EnrTensorboard, TestCallback


def train_val_test(args):
    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('GPU')

    if args['test']:
        data = MelData(task=args['task'], image_type=args['image_type'], pretrained=args['pretrained'],
                       dir_dict=args['dir_dict'], input_shape=args['input_shape'], dataset_frac=args['dataset_frac'])
        all_data = data.all_datasets(batch=128, no_image_type=args['no_image_type'], only_image=args['only_image'])
        model = tf.keras.models.load_model(args['dir_dict']['model_path'], compile=False)
        if args['task'] in ('ben_mal', '5cls'):
            calc_metrics(args=args, model=model, dataset=all_data['isic20_test'], dataset_type='isic20_test')
        calc_metrics(args=args, model=model, dataset=all_data['validation'], dataset_type='validation')
        calc_metrics(args=args, model=model, dataset=all_data['test'], dataset_type='test')
    else:
        os.makedirs(args['dir_dict']['logs'], exist_ok=True)
        os.makedirs(args['dir_dict']['trial'], exist_ok=True)
        with open(args['dir_dict']['hparams_logs'], 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', os.path.basename(args['dir_dict']['trial'])])
            [writer.writerow([key, str(args[key])]) for key in args.keys() if key != 'dir_dict']

        batch = args['batch_size'] * strategy.num_replicas_in_sync
        data = MelData(task=args['task'], image_type=args['image_type'], pretrained=args['pretrained'],
                       dir_dict=args['dir_dict'], input_shape=args['input_shape'], dataset_frac=args['dataset_frac'])
        all_data = data.all_datasets(batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
        data.logs()

        loss_fn = {'cxe': 'categorical_crossentropy', 'focal': binary_focal_loss(), 'custom': custom_loss(args['loss_frac']),
                   'perclass': PerClassWeightedCategoricalCrossentropy(args=args)}[args['loss_fn']]

        optimizer = {'adam': tf.keras.optimizers.Adam, 'adamax': tf.keras.optimizers.Adamax, 'nadam': tf.keras.optimizers.Nadam,
                     'ftrl': tf.keras.optimizers.Ftrl, 'rmsprop': tf.keras.optimizers.RMSprop, 'sgd': tf.keras.optimizers.SGD,
                     'adagrad': tf.keras.optimizers.Adagrad, 'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']]

        with strategy.scope():
            custom_model = model_struct(args=args)
            custom_model.compile(optimizer=optimizer(learning_rate=args['learning_rate'] * strategy.num_replicas_in_sync),
                                 loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])
        # --------------------------------------------------- Callbacks ---------------------------------------------- #
        with open(os.path.join(args['dir_dict']['trial'], 'model_summary.txt'), 'w') as f:
            custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

        callbacks = [ReduceLROnPlateau(factor=0.1, patience=10, verbose=args['verbose']),
                     EarlyStopping(patience=20, verbose=args['verbose']),
                     LaterCheckpoint(filepath=args['dir_dict']['model_path'], save_best_only=True, start_at=10, verbose=args['verbose']),
                     EnrTensorboard(val_data=data.get_dataset(mode='validation', batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image']),
                                    log_dir=args['dir_dict']['logs'], class_names=args['class_names']),
                     hp.KerasCallback(writer=args['dir_dict']['logs'], hparams={'pretrained': args['pretrained'], 'task':args['task'],
                                                                                'image_type':args['image_type'], 'image_size':args['image_size'],
                                                                                'only_image':args['only_image'], 'colour':args['colour'],
                                                                                'batch_size':args['batch_size'], 'learning_rate':args['learning_rate'],
                                                                                'optimizer':args['optimizer'], 'activation':args['activation'],
                                                                                'dropout':args['dropout'], 'epochs':args['epochs'],
                                                                                'conv_layers':args['conv_layers'], 'no_image_type':args['no_image_type']},
                                      trial_id=os.path.basename(args["dir_dict"]["trial"])),
                     TestCallback(args=args, isic20_test=data.get_dataset(mode='isic20_test', batch=batch),
                                  validation=data.get_dataset(mode='validation', batch=batch),
                                  test=data.get_dataset(mode='test', batch=batch))]

        train_1 = custom_model.fit(x=all_data['train'], validation_data=all_data['validation'], epochs=args['epochs'], callbacks=callbacks, verbose=args['verbose'])

# --------------------------------------------------- FINE TUNING ---------------------------------------------------- #
        n_epochs = len(train_1.history['loss'])
        args['learning_rate'] = 1e-6
        batch = 4 * strategy.num_replicas_in_sync
        all_data = data.all_datasets(batch=batch, no_image_type=args['no_image_type'], only_image=args['only_image'])
        args['dir_dict']['trial'] = os.path.join(args['dir_dict']['trial'], 'fine')
        args['dir_dict']['model_path'] = os.path.join(args['dir_dict']['trial'], 'model')
        os.makedirs(args['dir_dict']['trial'], exist_ok=True)

        def unfreeze_model(trained_model):
            for layer in trained_model.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            return trained_model

        with strategy.scope():
            if args['fine']:
                custom_model = tf.keras.models.load_model(args['load_model'], compile=False)
            custom_model = unfreeze_model(custom_model)
            custom_model.compile(optimizer=optimizer(learning_rate=args['learning_rate'] * strategy.num_replicas_in_sync), loss=loss_fn, metrics=[tfa.metrics.F1Score(num_classes=args['num_classes'], average='macro')])  # binary_focal_loss()

        with open(os.path.join(args['dir_dict']['trial'], 'fine_model_summary.txt'), 'w') as f:
            custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

        callbacks = [ReduceLROnPlateau(factor=0.1, patience=5, verbose=args['verbose']),
                     EarlyStopping(patience=10, verbose=args['verbose']),
                     LaterCheckpoint(filepath=args['dir_dict']['model_path'], save_best_only=True, start_at=0, verbose=args['verbose']),
                     EnrTensorboard(log_dir=args['dir_dict']['logs'], val_data=all_data['validation'], class_names=args['class_names']),
                     hp.KerasCallback(args['dir_dict']['logs'],
                                      hparams={'pretrained': args['pretrained'], 'task':args['task'],
                                               'image_type':args['image_type'], 'image_size':args['image_size'],
                                               'only_image':args['only_image'], 'batch_size':args['batch_size'],
                                               'learning_rate':args['learning_rate'], 'optimizer':args['optimizer'],
                                               'activation':args['activation'], 'dropout':args['dropout'], 'epochs':args['epochs'],
                                               'conv_layers':args['conv_layers'], 'no_image_type':args['no_image_type']},
                                      trial_id=args["dir_dict"]["trial"].split('/')[-1] + '_fine'),
                     TestCallback(args=args, isic20_test=data.get_dataset(mode='isic20_test', batch=batch),
                                  validation=data.get_dataset(mode='validation', batch=batch),
                                  test=data.get_dataset(mode='test', batch=batch))]

        custom_model.fit(x=all_data['train'], validation_data=all_data['validation'], callbacks=callbacks,
                         initial_epoch=n_epochs, epochs=n_epochs + 10, verbose=args['verbose'])
