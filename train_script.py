import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorboard.plugins.hparams import api as hp

from config import IMAGE_TYPE
from data_pipe import MelData
from metrics import calc_metrics
from model import model_fn
from callbacks import LaterCheckpoint, EnrTensorboard, TestCallback


def train_val_test(args):
    if args['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('gpu')
    args['replicas'] = strategy.num_replicas_in_sync
    data = MelData(args=args)
    args['train_data'] = data.get_dataset(mode='train')
    args['val_data'] = data.get_dataset(mode='val')
    args['test_data'] = data.get_dataset(mode='test')
    args['isic20_test'] = data.get_dataset(mode='isic20_test')
    if args['test'] or args['validate']:
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
        os.makedirs(args['dir_dict']['logs'], exist_ok=True)
        os.makedirs(args['dir_dict']['trial'], exist_ok=True)
        data = MelData(args=args)
        weights_per_class, weights_per_image_type, image_type_counts, class_counts = data.weights()
        with open(args['dir_dict']['hparams_logs'], 'w') as f:
            [f.write(': '.join([key.capitalize().rjust(len(max(args.keys(), key=len))), str(args[key])]) + '\n')
             for key in args.keys() if key not in ('dir_dict', 'hparams', 'train_data', 'val_data', 'test_data', 'isic20_test')]
            if not args['no_image_weights']:
                f.write('Weights per class\n')
                if args['image_type'] == 'both':
                    image_types = IMAGE_TYPE
                else:
                    image_types = [args['image_type']]
                for _image_type in image_types:
                    f.write(_image_type + '\n')
                    for _class in args['class_names']:
                        f.write(': '.join([_class, str(weights_per_image_type[_image_type] * weights_per_class[_class])+'\n']))

        optimizer = {'adam': tf.keras.optimizers.Adam, 'ftrl': tf.keras.optimizers.Ftrl,
                     'sgd': tf.keras.optimizers.SGD, 'rmsprop': tf.keras.optimizers.RMSprop,
                     'adadelta': tf.keras.optimizers.Adadelta, 'adagrad': tf.keras.optimizers.Adagrad,
                     'adamax': tf.keras.optimizers.Adamax, 'nadam': tf.keras.optimizers.Nadam}[args['optimizer']](learning_rate=args['learning_rate'] * args['replicas'])
        with strategy.scope():
            custom_model = model_fn(args=args)
            with open(os.path.join(args['dir_dict']['trial'], 'model_summary.txt'), 'w') as f:
                custom_model.summary(print_fn=lambda x: f.write(x + '\n'))

            custom_model.compile(optimizer=optimizer,
                                 loss='categorical_crossentropy',
                                 metrics=[AUC(multi_label=True)])
        # --------------------------------------------------- Callbacks ---------------------------------------------- #
        callbacks = [LaterCheckpoint(filepath=args['dir_dict']['model_path'], save_best_only=True, start_at=0),
                     EnrTensorboard(log_dir=args['dir_dict']['logs'], val_data=args['val_data'], class_names=args['class_names']),
                     ReduceLROnPlateau(factor=0.75, patience=10),
                     EarlyStopping(verbose=args['verbose'], patience=20),
                     hp.KerasCallback(args['dir_dict']['logs'], hparams={'pretrained': args['pretrained'], 'task':args['task'],
                                                                         'image_type':args['image_type'], 'image_size':args['image_size'],
                                                                         'only_image':args['only_image'], 'colour':args['colour'],
                                                                         'batch_size':args['batch_size'], 'learning_rate':args['learning_rate'],
                                                                         'optimizer':args['optimizer'], 'activation':args['activation'],
                                                                         'dropout':args['dropout'], 'epochs':args['epochs'],
                                                                         'layers':args['layers'], 'no_image_weights':args['no_image_weights'],
                                                                         'no_image_type':args['no_image_type']
                                                                         }, trial_id=os.path.basename(args["dir_dict"]["trial"])),
                     TestCallback(args=args)]
        # ------------------------------------------------- Train model ---------------------------------------------- #
        custom_model.fit(x=args['train_data'], epochs=args['epochs'],
                         validation_data=args['val_data'],
                         callbacks=callbacks, verbose=args['verbose'])
