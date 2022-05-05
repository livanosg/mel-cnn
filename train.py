from contextlib import redirect_stdout
import tensorflow as tf
import tensorflow_addons as tfa
from features_def import TASK_CLASSES
from custom_metrics import GeometricMean
from custom_callbacks import EnrTensorboard
from custom_losses import losses


def train_fn(args: dict, dirs: dict, data, model, strategy):
    """Setup and run training stage"""
    loss = losses(args)[args['loss_fn']]

    with strategy.scope():
        optimizer = {'adam': tf.keras.optimizers.Adam,
                     'adamax': tf.keras.optimizers.Adamax,
                     'nadam': tf.keras.optimizers.Nadam,
                     'ftrl': tf.keras.optimizers.Ftrl,
                     'rmsprop': tf.keras.optimizers.RMSprop,
                     'sgd': tf.keras.optimizers.SGD,
                     'adagrad': tf.keras.optimizers.Adagrad,
                     'adadelta': tf.keras.optimizers.Adadelta}[args['optimizer']](learning_rate=args['learning_rate'] * args['gpus'])

        model.compile(loss=loss, optimizer=optimizer,
                      metrics=[tfa.metrics.F1Score(num_classes=len(TASK_CLASSES[args['task']]),
                                                   average='macro', name='f1'),
                               GeometricMean()])
    with redirect_stdout(open(dirs['model_summary'], 'w', encoding='utf-8')):
        model.summary(show_trainable=True)

    model.fit(x=data.get_dataset('train'), validation_data=data.get_dataset('validation'), epochs=args['epochs'],
              callbacks=[tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'], separator=',', append=True),
                         tf.keras.callbacks.EarlyStopping(monitor='val_gmean_macro', mode='max', verbose=1,
                                                          patience=args['early_stop'], restore_best_weights=True),
                         tf.keras.callbacks.CSVLogger(filename=dirs['train_logs'], separator=',', append=True),
                         EnrTensorboard(val_data=data.get_dataset('validation'), log_dir=dirs['logs'],
                                        class_names=TASK_CLASSES[args['task']])
                         ]
              )
    model.compile(optimizer=optimizer, loss=None)
    model.save(filepath=dirs['save_path'])
    return model
